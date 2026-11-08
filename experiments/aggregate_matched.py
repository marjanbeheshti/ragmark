#!/usr/bin/env python3
"""
Aggregate multi-seed / multi-model matched_poisonedrag runs
===========================================================
Reads every results/matched_poisonedrag_s*.json, groups by LLM model
(retrieval-only runs grouped as 'retrieval'), and prints mean +- std
across seeds for recall@5 (A/B/C), poison@5, and LLM correct/tricked
-- the numbers Table `tab:poisonedrag` needs with seed variance.

Usage: python experiments/aggregate_matched.py
"""
import glob
import json
import os
import re
import sys

import numpy as np


def frac(s):
    a, b = s.split("/")
    return 100.0 * int(a) / int(b)


def main():
    files = sorted(glob.glob("results/matched_poisonedrag_s*.json"))
    if not files:
        sys.exit("No results/matched_poisonedrag_s*.json found -- run "
                 "matched_poisonedrag.py first.")
    groups = {}
    for f in files:
        r = json.load(open(f))
        model = r.get("llm_model", "retrieval")
        groups.setdefault(model, []).append((r.get("seed", "?"), r))

    for model, runs in groups.items():
        seeds = [s for s, _ in runs]
        print("=" * 64)
        print(f"MODEL: {model}   seeds: {seeds}  (n={len(runs)})")
        print("=" * 64)

        def agg(fn, label):
            vals = np.array([fn(r) for _, r in runs], dtype=float)
            print(f"  {label:<34s} {vals.mean():6.1f} +- "
                  f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:4.1f}"
                  f"   {[round(v,1) for v in vals]}")
            return vals

        print("-- retrieval --")
        agg(lambda r: frac(r["recall_A"]), "Recall@5 (A) clean %")
        agg(lambda r: frac(r["recall_B"]), "Recall@5 (B) attacked %")
        agg(lambda r: frac(r["recall_C"]), "Recall@5 (C) RagMark %")
        agg(lambda r: r["poison_at5"], "poison@5 (B)")
        agg(lambda r: frac(r["poison_rejected"]), "poison rejected %")
        agg(lambda r: frac(r["self_check"]), "self-check %")

        if model != "retrieval":
            print("-- LLM --")
            for c, name in [("A", "clean"), ("B", "attacked"),
                            ("C", "RagMark")]:
                agg(lambda r, c=c: 100 * r[f"llm_{c}"]["correct"] /
                    max(r[f"llm_{c}"]["total"], 1),
                    f"({c}) {name}: correct %")
                agg(lambda r, c=c: 100 * r[f"llm_{c}"]["tricked"] /
                    max(r[f"llm_{c}"]["total"], 1),
                    f"({c}) {name}: tricked %")
        print()

    print("Paste the qwen3:8b block into tab:poisonedrag as mean+-std;")
    print("cite the second model's (C) row in one sentence as a")
    print("cross-LLM check.")


if __name__ == "__main__":
    main()
