"""Estimate the coupling factor g from a CST S21 anticrossing colorplot.

Pipeline:
  1. parse_cst_txt        - read CST ASCII S21 export (parametric sweep)
  2. extract_two_branches - pick the two anticrossing dips per slice
  3. fit_coupling         - fit f_+/f_- = (f1+f2)/2 +/- sqrt(((f1-f2)/2)^2 + g^2)
  4. plot_result          - colormap + overlaid extracted points and fit

Run `python coupling_fit.py --help` for CLI usage.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Parsing CST ASCII S21 exports
# ---------------------------------------------------------------------------

_PARAM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_NUMERIC_LINE_RE = re.compile(
    r"^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$"
)


@dataclass
class S21Data:
    params: np.ndarray         # shape (P,)
    freqs: np.ndarray          # shape (F,) - common frequency axis
    s21: np.ndarray            # shape (P, F) - magnitude (dB if input was dB)
    param_name: str | None
    freq_unit: str | None
    s21_unit: str | None


def parse_cst_txt(path: str | Path, param_name: str | None = None) -> S21Data:
    """Parse a CST ASCII S21 export.

    Handles two layouts:
      (a) Block-per-parameter: each block has a header containing "name=value"
          followed by two numeric columns (frequency, S21).
      (b) Wide table: first numeric column is frequency, remaining columns are
          one S21 trace per parameter value (parameter value taken from the
          column header `... (name=value) ...`).
    """
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()

    blocks = _split_into_blocks(lines)
    parsed_blocks = [_parse_block(b, param_name) for b in blocks]
    parsed_blocks = [b for b in parsed_blocks if b is not None and len(b[1]) > 1]

    if len(parsed_blocks) >= 2:
        return _assemble_block_data(parsed_blocks)

    wide = _try_parse_wide(lines, param_name)
    if wide is not None:
        return wide

    raise ValueError(
        f"Could not parse {path}: expected either multiple parameter blocks "
        f"or a wide table with parameter values in column headers."
    )


def _split_into_blocks(lines: list[str]) -> list[list[str]]:
    """Split on blank lines and on 'Parameters:' markers; drop pure-decoration
    lines (sequences of -, =, #, *) so they don't fragment a block."""
    cleaned = [ln for ln in lines if not (ln.strip() and set(ln.strip()) <= set("-=#*"))]

    blocks: list[list[str]] = []
    current: list[str] = []
    for ln in cleaned:
        stripped = ln.strip()
        starts_new = bool(re.match(r"\s*Parameters\s*[:=]", ln, re.IGNORECASE))
        if not stripped:
            if current:
                blocks.append(current)
                current = []
            continue
        if starts_new and current and any(_NUMERIC_LINE_RE.match(x) for x in current):
            blocks.append(current)
            current = []
        current.append(ln)
    if current:
        blocks.append(current)
    return blocks


def _parse_block(
    block: list[str], param_name: str | None
) -> tuple[float | None, np.ndarray, str | None, str | None] | None:
    param_value: float | None = None
    chosen_param_name = param_name
    freq_unit: str | None = None
    s21_unit: str | None = None
    data: list[tuple[float, float]] = []

    for ln in block:
        if _NUMERIC_LINE_RE.match(ln):
            a, b = ln.split()
            data.append((float(a), float(b)))
            continue
        for m in _PARAM_RE.finditer(ln):
            name, val = m.group(1), m.group(2)
            if name.lower() in {"e", "ghz", "mhz", "khz", "hz", "db"}:
                continue
            if param_name is None or name == param_name:
                try:
                    param_value = float(val)
                    chosen_param_name = name
                except ValueError:
                    pass
        fu = re.search(r"Frequency\s*/\s*(\w+)", ln, re.IGNORECASE)
        if fu:
            freq_unit = fu.group(1)
        su = re.search(r"S\d?\d?\s*/\s*([\w]+)", ln, re.IGNORECASE)
        if su and "freq" not in ln.lower():
            s21_unit = su.group(1)

    if not data:
        return None
    arr = np.asarray(data, dtype=float)
    return param_value, arr, freq_unit, s21_unit, chosen_param_name  # type: ignore[return-value]


def _assemble_block_data(parsed_blocks) -> S21Data:
    params, arrays, freq_units, s21_units, names = zip(
        *[(p, a, fu, su, nm) for (p, a, fu, su, nm) in parsed_blocks]
    )
    if any(p is None for p in params):
        params = tuple(float(i) if p is None else p for i, p in enumerate(params))

    freq_min = max(a[:, 0].min() for a in arrays)
    freq_max = min(a[:, 0].max() for a in arrays)
    n = max(len(a) for a in arrays)
    common_f = np.linspace(freq_min, freq_max, n)

    s21 = np.empty((len(arrays), n), dtype=float)
    for i, a in enumerate(arrays):
        order = np.argsort(a[:, 0])
        s21[i] = np.interp(common_f, a[order, 0], a[order, 1])

    p_arr = np.asarray(params, dtype=float)
    order = np.argsort(p_arr)
    return S21Data(
        params=p_arr[order],
        freqs=common_f,
        s21=s21[order],
        param_name=next((n for n in names if n), None),
        freq_unit=next((u for u in freq_units if u), None),
        s21_unit=next((u for u in s21_units if u), None),
    )


def _try_parse_wide(lines: list[str], param_name: str | None) -> S21Data | None:
    header_idx = None
    for i, ln in enumerate(lines):
        if _PARAM_RE.search(ln) and ln.count("=") >= 2:
            header_idx = i
            break
    if header_idx is None:
        return None

    header = lines[header_idx]
    matches = list(_PARAM_RE.finditer(header))
    name_counts: dict[str, int] = {}
    for m in matches:
        name_counts[m.group(1)] = name_counts.get(m.group(1), 0) + 1
    if param_name is None:
        param_name = max(name_counts, key=name_counts.get)
    params = [float(m.group(2)) for m in matches if m.group(1) == param_name]
    if len(params) < 2:
        return None

    data_rows: list[list[float]] = []
    for ln in lines[header_idx + 1 :]:
        toks = ln.split()
        try:
            row = [float(t) for t in toks]
        except ValueError:
            continue
        if len(row) >= 1 + len(params):
            data_rows.append(row[: 1 + len(params)])
    if len(data_rows) < 2:
        return None

    arr = np.asarray(data_rows, dtype=float)
    freqs = arr[:, 0]
    s21 = arr[:, 1:].T
    order = np.argsort(np.asarray(params))
    return S21Data(
        params=np.asarray(params)[order],
        freqs=freqs,
        s21=s21[order],
        param_name=param_name,
        freq_unit=None,
        s21_unit=None,
    )


# ---------------------------------------------------------------------------
# Branch extraction
# ---------------------------------------------------------------------------

def extract_two_branches(
    data: S21Data,
    freq_window: tuple[float, float] | None = None,
    prominence_db: float = 1.0,
    min_separation: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the two anticrossing dips per parameter slice.

    Strategy per slice:
      - restrict to ``freq_window`` if given
      - invert S21 (treat dips as peaks)
      - find_peaks with ``prominence_db`` and optional ``distance``
      - keep the two most prominent peaks
      - sort into (lower, upper) frequency

    Returns ``(params_kept, f_lower, f_upper)``. Slices that yielded fewer than
    two qualifying peaks are dropped.
    """
    if freq_window is not None:
        mask = (data.freqs >= freq_window[0]) & (data.freqs <= freq_window[1])
    else:
        mask = np.ones_like(data.freqs, dtype=bool)
    f = data.freqs[mask]

    distance = None
    if min_separation is not None:
        df = float(np.median(np.diff(f)))
        if df > 0:
            distance = max(1, int(round(min_separation / df)))

    p_kept: list[float] = []
    f_lo: list[float] = []
    f_hi: list[float] = []
    for i, p in enumerate(data.params):
        trace = -data.s21[i, mask]
        peaks, props = find_peaks(trace, prominence=prominence_db, distance=distance)
        if len(peaks) < 2:
            continue
        order = np.argsort(props["prominences"])[::-1]
        top2 = np.sort(peaks[order[:2]])
        p_kept.append(p)
        f_lo.append(f[top2[0]])
        f_hi.append(f[top2[1]])
    return np.asarray(p_kept), np.asarray(f_lo), np.asarray(f_hi)


# ---------------------------------------------------------------------------
# Anticrossing fit
# ---------------------------------------------------------------------------

def anticrossing_lower(p, f_a0, s_a, f_b0, s_b, g, p0):
    fa = f_a0 + s_a * (p - p0)
    fb = f_b0 + s_b * (p - p0)
    return 0.5 * (fa + fb) - np.sqrt(0.25 * (fa - fb) ** 2 + g ** 2)


def anticrossing_upper(p, f_a0, s_a, f_b0, s_b, g, p0):
    fa = f_a0 + s_a * (p - p0)
    fb = f_b0 + s_b * (p - p0)
    return 0.5 * (fa + fb) + np.sqrt(0.25 * (fa - fb) ** 2 + g ** 2)


@dataclass
class FitResult:
    f_a0: float
    s_a: float
    f_b0: float
    s_b: float
    g: float
    p0: float
    perr: dict[str, float]

    def lower(self, p):
        return anticrossing_lower(p, self.f_a0, self.s_a, self.f_b0, self.s_b, self.g, self.p0)

    def upper(self, p):
        return anticrossing_upper(p, self.f_a0, self.s_a, self.f_b0, self.s_b, self.g, self.p0)


def fit_coupling(
    params: np.ndarray,
    f_lower: np.ndarray,
    f_upper: np.ndarray,
) -> FitResult:
    """Joint fit of both branches in the identifiable (sum, diff) parametrization.

    Returns a FitResult with the coupling g (same units as freq). The per-mode
    fields (f_a0, s_a, f_b0, s_b) are reconstructed with the convention that
    mode 'b' has the larger slope magnitude (the tuned mode).
    """
    p0_anchor = float(np.mean(params))
    gap = f_upper - f_lower
    g0 = float(np.min(gap) / 2.0)
    sum_mid = 0.5 * (f_lower + f_upper)
    diff_outer = (f_upper[-1] - f_lower[-1]) - (f_upper[0] - f_lower[0])
    span = params[-1] - params[0]
    m0_init = float(np.mean(sum_mid))
    ms_init = float((sum_mid[-1] - sum_mid[0]) / (span + 1e-30))
    d0_init = 0.0
    ds_init = float(diff_outer / (span + 1e-30)) if span != 0 else 1.0

    def model(p_concat, m0, ms, d0, ds, g):
        half = len(p_concat) // 2
        p_lo, p_hi = p_concat[:half], p_concat[half:]
        sum_lo = m0 + ms * (p_lo - p0_anchor)
        diff_lo = d0 + ds * (p_lo - p0_anchor)
        sum_hi = m0 + ms * (p_hi - p0_anchor)
        diff_hi = d0 + ds * (p_hi - p0_anchor)
        lo = sum_lo - np.sqrt(0.25 * diff_lo ** 2 + g ** 2)
        hi = sum_hi + np.sqrt(0.25 * diff_hi ** 2 + g ** 2)
        return np.concatenate([lo, hi])

    x = np.concatenate([params, params])
    y = np.concatenate([f_lower, f_upper])
    p0 = [m0_init, ms_init, d0_init, ds_init, g0]

    popt, pcov = curve_fit(model, x, y, p0=p0, maxfev=20000)
    perr_red = np.sqrt(np.diag(pcov))
    m0, ms, d0, ds, g = popt
    g = abs(g)

    f_a0 = m0 + 0.5 * d0
    s_a = ms + 0.5 * ds
    f_b0 = m0 - 0.5 * d0
    s_b = ms - 0.5 * ds
    if abs(s_a) > abs(s_b):
        f_a0, f_b0 = f_b0, f_a0
        s_a, s_b = s_b, s_a

    perr = {
        "f_a0": float(np.hypot(perr_red[0], 0.5 * perr_red[2])),
        "s_a":  float(np.hypot(perr_red[1], 0.5 * perr_red[3])),
        "f_b0": float(np.hypot(perr_red[0], 0.5 * perr_red[2])),
        "s_b":  float(np.hypot(perr_red[1], 0.5 * perr_red[3])),
        "g":    float(perr_red[4]),
        "p0":   0.0,
    }
    return FitResult(
        f_a0=f_a0, s_a=s_a, f_b0=f_b0, s_b=s_b, g=g, p0=p0_anchor, perr=perr,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_result(
    data: S21Data,
    params_pts: np.ndarray,
    f_lo: np.ndarray,
    f_hi: np.ndarray,
    fit: FitResult,
    out: str | Path | None = None,
    show: bool = True,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    pm = ax.pcolormesh(data.params, data.freqs, data.s21.T, shading="auto", cmap="viridis")
    cb = fig.colorbar(pm, ax=ax)
    cb.set_label(f"|S21| / {data.s21_unit or 'dB'}")

    ax.plot(params_pts, f_lo, "o", ms=4, color="white", mec="black", label="extracted dips")
    ax.plot(params_pts, f_hi, "o", ms=4, color="white", mec="black")

    p_dense = np.linspace(data.params.min(), data.params.max(), 400)
    ax.plot(p_dense, fit.lower(p_dense), "-", color="red", lw=1.5, label="fit")
    ax.plot(p_dense, fit.upper(p_dense), "-", color="red", lw=1.5)

    ax.set_xlabel(data.param_name or "parameter")
    ax.set_ylabel(f"frequency / {data.freq_unit or 'GHz'}")
    ax.set_title(f"Anticrossing fit: g = {fit.g:.4g} +/- {fit.perr['g']:.2g}")
    ax.legend(loc="best")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="CST ASCII .txt export of |S21|")
    ap.add_argument("--param", default=None, help="parameter name (auto-detected if omitted)")
    ap.add_argument("--fmin", type=float, default=None, help="frequency window lower bound")
    ap.add_argument("--fmax", type=float, default=None, help="frequency window upper bound")
    ap.add_argument("--prominence", type=float, default=1.0,
                    help="minimum dip prominence in dB (default 1.0)")
    ap.add_argument("--min-separation", type=float, default=None,
                    help="minimum dip separation in frequency units")
    ap.add_argument("--save", default=None, help="path to save the plot (e.g. fit.png)")
    ap.add_argument("--no-show", action="store_true", help="don't open the plot window")
    args = ap.parse_args(argv)

    data = parse_cst_txt(args.input, param_name=args.param)
    print(f"Loaded {data.s21.shape[0]} slices x {data.s21.shape[1]} frequencies")
    print(f"  param '{data.param_name}' range: {data.params.min():.4g} .. {data.params.max():.4g}")
    print(f"  freq range: {data.freqs.min():.4g} .. {data.freqs.max():.4g} {data.freq_unit or ''}")

    window = None
    if args.fmin is not None or args.fmax is not None:
        window = (args.fmin if args.fmin is not None else data.freqs.min(),
                  args.fmax if args.fmax is not None else data.freqs.max())

    p_pts, f_lo, f_hi = extract_two_branches(
        data, freq_window=window,
        prominence_db=args.prominence,
        min_separation=args.min_separation,
    )
    if len(p_pts) < 4:
        raise SystemExit(
            f"Only {len(p_pts)} slices yielded two dips. Adjust --fmin/--fmax/--prominence."
        )
    print(f"Extracted two-dip pairs from {len(p_pts)} / {len(data.params)} slices")

    fit = fit_coupling(p_pts, f_lo, f_hi)
    fu = data.freq_unit or "GHz"
    print("\nFit results:")
    print(f"  g       = {fit.g:.6g} +/- {fit.perr['g']:.2g} {fu}")
    print(f"  2g (min splitting) = {2*fit.g:.6g} {fu}")
    print(f"  f_a0    = {fit.f_a0:.6g} {fu}   slope_a = {fit.s_a:.4g} {fu}/unit")
    print(f"  f_b0    = {fit.f_b0:.6g} {fu}   slope_b = {fit.s_b:.4g} {fu}/unit")
    print(f"  p0      = {fit.p0:.6g}")

    plot_result(data, p_pts, f_lo, f_hi, fit, out=args.save, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
