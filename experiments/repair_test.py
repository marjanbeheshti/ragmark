#!/usr/bin/env python3
"""
Metadata Repair (Re-Signing) Test (Walkthrough Sec. 6.9)
========================================================
Strict mode rejects vectors without metadata -- so how is LEGITIMATE
metadata loss managed? By repair, not by a verification loophole: the
key holder can re-create the whole signed blob from the vector itself
(ragmark_hardened.repair_metadata). This experiment validates the
maintenance path and its refusal behavior:

  (A) Strip 100 legitimate blobs -> strict mode rejects all (as it
      must) -> repair all 100 -> strict mode accepts all again.
  (B) Negative control: injected (foreign) vectors cannot be
      repaired -- the E_I gate refuses to bless them.
  (C) Honest caveat, demonstrated: a vector modified INSIDE E_I's
      blind spots (free-position permutation) while unsigned WILL be
      blessed by repair -- repair certifies from repair time forward,
      not retroactively. This is why loss handling is
      quarantine -> verify -> re-sign with a short window, not casual.

Writes results/repair_test.json
Usage: python experiments/repair_test.py
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import ragmark as R
import ragmark_hardened as RH
import permutation_attack as P
from sklearn.decomposition import PCA

N_DOCS = 500
KEY = R.MASTER_KEY
SEED = 2026


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"date": datetime.now().isoformat(),
           "test": "metadata repair (re-signing)", "seed": SEED}

    print("=" * 62)
    print("Metadata Repair (Re-Signing) Test (Sec. 6.9)")
    print("=" * 62)

    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    print(f"\n[SETUP] Loading corpus ({N_DOCS} docs)...")
    ds = load_dataset("rajpurkar/squad", split="train")
    seen, texts = set(), []
    for r in ds:
        if r["context"] not in seen and len(texts) < N_DOCS:
            seen.add(r["context"]); texts.append(r["context"])
    model = SentenceTransformer("all-mpnet-base-v2")
    E = np.asarray(model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=True))
    d = E.shape[1]
    pca = PCA(n_components=min(d, len(E))).fit(E)
    U, mu = pca.components_, pca.mean_
    pc_dim = U.shape[0]
    e1_band, e0_band, e0_ppb = R.compute_bands(d)
    print(f"  d={d}, E_A={e1_band}, E_I={e0_band}, ppb={e0_ppb}")

    print("  Watermarking (hardened)...")
    wm, metas = [], []
    for i in range(len(E)):
        w_raw, meta = RH.watermark_hardened(E[i], i % 8, U, mu,
                                            e1_band, e0_band, e0_ppb,
                                            doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta)
    wm = np.asarray(wm)

    def strict_pass(v, meta):
        return RH.verify_hardened(v, U, mu, e0_band, e0_ppb, meta=meta,
                                  key=KEY,
                                  expected_doc_id=(meta["doc_id"]
                                                   if meta else None)
                                  )["verdict"] == "PASS"

    # =================================================================
    # (A) strip -> reject -> repair -> accept
    # =================================================================
    print("\n[A] Strip 100 blobs, repair, re-verify")
    idx = rng.choice(len(wm), size=100, replace=False)
    stripped_reject = sum(not strict_pass(wm[i], None) for i in idx)
    repaired = 0
    repaired_pass = 0
    sid_ok = 0
    statuses = {}
    for i in idx:
        meta2, status = RH.repair_metadata(wm[i], int(i), U, mu,
                                           e1_band, e0_band, e0_ppb, KEY)
        statuses[status] = statuses.get(status, 0) + 1
        if status != "REPAIRED":
            continue
        repaired += 1
        repaired_pass += strict_pass(wm[i], meta2)
        sid_ok += (meta2["sid"] == metas[i]["sid"])
    print(f"  Stripped vectors rejected (strict):  {stripped_reject}/100  "
          f"(expect 100)")
    print(f"  Repaired:                            {repaired}/100  "
          f"{statuses}")
    print(f"  Repaired vectors PASS (hardened):    {repaired_pass}/"
          f"{repaired}")
    print(f"  Recovered source ID correct:         {sid_ok}/{repaired}")
    res["A_stripped_reject"] = f"{stripped_reject}/100"
    res["A_repaired"] = f"{repaired}/100"
    res["A_repaired_pass"] = f"{repaired_pass}/{repaired}"
    res["A_sid_ok"] = f"{sid_ok}/{repaired}"
    res["A_statuses"] = statuses

    # =================================================================
    # (B) negative control: foreign vectors are refused
    # =================================================================
    print("\n[B] Negative control: repair refused on foreign vectors")
    refused = 0
    for _ in range(100):
        fake = rng.standard_normal(d)
        fake = fake / np.linalg.norm(fake)
        meta2, status = RH.repair_metadata(fake, 999, U, mu,
                                           e1_band, e0_band, e0_ppb, KEY)
        refused += (status == "REJECTED_EI")
    print(f"  Foreign vectors refused:             {refused}/100  "
          f"(expect 100)")
    res["B_refused"] = f"{refused}/100"

    # =================================================================
    # (C) the honest caveat: blind-spot edits during the unsigned
    #     window ARE blessed by repair
    # =================================================================
    print("\n[C] Caveat: permuted-while-unsigned vectors get blessed")
    blessed = 0
    for i in idx[:50]:
        v_pc = (wm[i] - mu) @ U.T
        free = P.free_positions(pc_dim, e1_band, e0_band,
                                metas[i]["sel"])
        a_pc = P.random_permutation(v_pc, free, rng)
        a_raw = a_pc @ U + mu
        meta2, status = RH.repair_metadata(a_raw, int(i), U, mu,
                                           e1_band, e0_band, e0_ppb, KEY)
        blessed += (status == "REPAIRED"
                    and strict_pass(a_raw, meta2))
    print(f"  Permuted vectors blessed by repair:  {blessed}/50")
    print("  (Expected high --- repair certifies from repair time")
    print("   forward; it cannot see edits inside E_I's blind spots")
    print("   made while the vector was unsigned. Hence: short")
    print("   quarantine windows, repair from backup-verified state.)")
    res["C_blessed"] = f"{blessed}/50"

    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  strip->reject {stripped_reject}/100, repair {repaired}/100, "
          f"re-verify {repaired_pass}/{repaired}, sid {sid_ok}/{repaired}")
    print(f"  foreign refused {refused}/100; "
          f"unsigned-window caveat: {blessed}/50 blessed (documented)")
    print("=" * 62)
    with open("results/repair_test.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/repair_test.json")


if __name__ == "__main__":
    main()
