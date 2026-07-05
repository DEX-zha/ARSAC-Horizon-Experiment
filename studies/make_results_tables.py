"""Generates the README/markdown and paper LaTeX tables from benchmark_final.csv.

No hand-copied numbers (Plan V2 Phase 4): run after studies/benchmark_final.py
completes. Writes outputs/benchmark_final_table.md and
outputs/benchmark_final_table.tex and prints both.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

CSV_PATH = os.path.join("outputs", "benchmark_final.csv")
MD_PATH = os.path.join("outputs", "benchmark_final_table.md")
TEX_PATH = os.path.join("outputs", "benchmark_final_table.tex")


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"missing {CSV_PATH}; run studies/benchmark_final.py first")
    groups = defaultdict(list)
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            groups[(r["dataset"], r["model"])].append(r)

    def q(rows, key, fn):
        vals = np.array([float(r[key]) for r in rows if r[key] not in ("", "None")])
        return fn(vals) if vals.size else float("nan")

    md = [
        "| System | Model | Seeds | Coverage med [min] | Tightness med | Slack p90 med | p_sat med | H_w med (steps) | Hmax |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    tex = [
        r"\begin{tabular}{l l r r r r r r r}",
        r"\toprule",
        r"System & Model & Seeds & CovMed & CovMin & TightMed & SlackP90 & pSatMed & $H_w$ med \\",
        r"\midrule",
    ]
    for (dataset, model) in sorted(groups):
        rows = groups[(dataset, model)]
        cov_med = q(rows, "coverage_test", np.median)
        cov_min = q(rows, "coverage_test", np.min)
        tight = q(rows, "tightness_ratio", np.median)
        slack = q(rows, "slack_p90", np.median)
        psat = q(rows, "p_sat_test", np.median)
        hmed = q(rows, "horizon_window_median", np.median)
        hmax = int(q(rows, "horizon_max", np.median))
        md.append(
            f"| {dataset} | {model} | {len(rows)} | {cov_med:.3f} [{cov_min:.3f}] "
            f"| {tight:.3f} | {slack:.1f} | {psat:.3f} | {hmed:.0f} | {hmax} |"
        )
        tex.append(
            f"{dataset.replace('_', chr(92) + '_')} & {model} & {len(rows)} & "
            f"{cov_med:.3f} & {cov_min:.3f} & {tight:.3f} & {slack:.1f} & "
            f"{psat:.3f} & {hmed:.0f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs("outputs", exist_ok=True)
    with open(MD_PATH, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(TEX_PATH, "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))
    print()
    print("\n".join(tex))


if __name__ == "__main__":
    main()
