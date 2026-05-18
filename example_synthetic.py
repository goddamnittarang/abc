"""Generate a synthetic anticrossing S21 colorplot and recover g.

Run:  python example_synthetic.py

Writes synthetic_cst_s21.txt in CST-like ASCII block format, then runs the
full extract + fit pipeline and prints the recovered coupling factor.
"""

from __future__ import annotations

import numpy as np

from coupling_fit import (
    parse_cst_txt,
    extract_two_branches,
    fit_coupling,
    plot_result,
)


TRUE_G = 0.020  # GHz - what we want to recover
F_RES = 5.000   # GHz - flat resonator mode
F_DSRR_AT_P0 = 5.000
SLOPE_DSRR = 1.0  # GHz per unit of sweep parameter
P0_TRUE = 0.0


def lorentzian_db(f, f0, kappa, depth_db=-20.0):
    return depth_db / (1.0 + ((f - f0) / (kappa / 2.0)) ** 2)


def true_modes(p):
    fa = F_RES + 0.0 * p
    fb = F_DSRR_AT_P0 + SLOPE_DSRR * (p - P0_TRUE)
    mean = 0.5 * (fa + fb)
    half = np.sqrt(0.25 * (fa - fb) ** 2 + TRUE_G ** 2)
    return mean - half, mean + half


def write_cst_like(path: str, params, freqs, s21_db):
    sep = "-" * 70 + "\n"
    with open(path, "w") as fh:
        for i, p in enumerate(params):
            fh.write(sep)
            fh.write(f"Parameters: sweep={p:.6g};\n")
            fh.write(sep)
            fh.write("   Frequency / GHz       S21 / dB\n")
            for f, s in zip(freqs, s21_db[i]):
                fh.write(f"   {f:.6e}     {s:.6e}\n")
            fh.write("\n")


def main():
    params = np.linspace(-0.10, 0.10, 41)
    freqs = np.linspace(4.90, 5.10, 801)
    s21 = np.zeros((len(params), len(freqs)))
    kappa_res, kappa_dsrr = 0.004, 0.006
    rng = np.random.default_rng(0)
    for i, p in enumerate(params):
        f_lo, f_hi = true_modes(p)
        trace = (
            lorentzian_db(freqs, f_lo, kappa_res, depth_db=-25.0)
            + lorentzian_db(freqs, f_hi, kappa_dsrr, depth_db=-20.0)
        )
        trace += rng.normal(0.0, 0.15, size=trace.shape)
        s21[i] = trace

    path = "synthetic_cst_s21.txt"
    write_cst_like(path, params, freqs, s21)
    print(f"Wrote {path}")

    data = parse_cst_txt(path)
    print(f"Parsed {data.s21.shape}, param={data.param_name}")

    p_pts, f_lo, f_hi = extract_two_branches(data, prominence_db=0.5)
    print(f"Extracted dips from {len(p_pts)}/{len(data.params)} slices")

    fit = fit_coupling(p_pts, f_lo, f_hi)
    print(f"\nTrue g  = {TRUE_G:.4g} GHz")
    print(f"Fit g   = {fit.g:.4g} +/- {fit.perr['g']:.2g} GHz")
    print(f"Fit 2g  = {2*fit.g:.4g} GHz")

    plot_result(data, p_pts, f_lo, f_hi, fit, out="synthetic_fit.png", show=False)
    print("Saved synthetic_fit.png")


if __name__ == "__main__":
    main()
