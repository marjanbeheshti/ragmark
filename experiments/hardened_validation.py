#!/usr/bin/env python3
"""
Hardened-Verifier Validation: every attack, one table
=====================================================
The definitive closure run. Re-executes EVERY attack in the paper's
security section against the hardened verifier (h_v + strict mode +
doc_id binding + manifest), plus the legitimate baseline, and prints a
single paper-ready table. This replaces the walkthrough's asserted
"0/500 expected" for sub-resolution with a measured number, and
confirms no attack survives the fix while no legitimate vector is
falsely rejected.

Rows:
  legit baseline           -> expect ~500/500 PASS (no false rejects)
  sub-resolution nudge     -> expect 0 PASS   (h_v mismatch)
  sub-res nudge + del meta -> expect 0 PASS   (strict)
  free-position permutation-> expect 0 PASS   (h_v mismatch)
  permutation + del meta   -> expect 0 PASS   (strict)
  rebinding (no meta)      -> expect 0 PASS   (strict)
  index rebind (meta kept) -> expect 0 PASS   (doc_id binding)
  revoked-doc replay       -> expect 0 PASS   (manifest)
  flooding replicas        -> PASS by design (byte-genuine);
                              dedup restores top-5 (control column)

Writes results/hardened_validation.json
Usage: python experiments/hardened_validation.py
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
import sub_resolution_attack as S
import permutation_attack as P
from sklearn.decomposition import PCA

N_DOCS = 500
KEY = R.MASTER_KEY
SEED = 2026


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"date": datetime.now().isoformat(),
           "test": "all attacks vs hardened verifier"}

    print("=" * 66)
    print("Hardened-Verifier Validation (all attacks, one table)")
    print("=" * 66)

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
    print(f"  d={d}, pc_dim={pc_dim}, E_A={e1_band}, E_I={e0_band}, "
          f"ppb={e0_ppb}")

    print("  Watermarking (hardened)...")
    wm, metas, doc_ids = [], [], []
    for i in range(len(E)):
        w_raw, meta = RH.watermark_hardened(E[i], i % 8, U, mu,
                                            e1_band, e0_band, e0_ppb,
                                            doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta); doc_ids.append(i)
    wm = np.asarray(wm)
    wm_pc = np.asarray([S.to_pc(wm[i], U, mu) for i in range(len(wm))])
    manifest = RH.build_manifest(doc_ids, epoch=1, key=KEY)

    def H(v_raw, i, meta=True, man=None):
        return RH.verify_hardened(
            v_raw, U, mu, e0_band, e0_ppb,
            meta=(metas[i] if meta else None), key=KEY,
            expected_doc_id=(metas[i]["doc_id"] if meta else None),
            manifest=man)["verdict"] == "PASS"

    rows = []

    # -- legit baseline (no false rejects) ---------------------------
    n = sum(H(wm[i], i, man=manifest) for i in range(len(wm)))
    rows.append(("legitimate vectors", f"{n}/{len(wm)}", "expect ~all"))

    # -- sub-resolution nudge (max-drift, worst case) ----------------
    n = nd = 0
    for i in range(len(wm)):
        lo, hi = S.bucket_edges(wm_pc[i])
        far_up = (hi - wm_pc[i]) >= (wm_pc[i] - lo)
        direction = np.where(far_up, 1.0, -1.0)
        a_pc = S.within_bucket_attack(wm_pc[i], direction, metas[i],
                                      e0_band, e0_ppb, KEY)
        a_raw = S.to_raw(a_pc, U, mu)
        n += H(a_raw, i)                       # meta present -> h_v
        nd += H(a_raw, i, meta=False)          # meta deleted -> strict
    rows.append(("sub-resolution nudge", f"{n}/{len(wm)}", "expect 0"))
    rows.append(("sub-res nudge + delete meta", f"{nd}/{len(wm)}",
                 "expect 0"))

    # -- free-position permutation -----------------------------------
    n = nd = 0
    for i in range(len(wm)):
        free = P.free_positions(pc_dim, e1_band, e0_band, metas[i]["sel"])
        a_pc = P.random_permutation(wm_pc[i], free, rng)
        a_raw = S.to_raw(a_pc, U, mu)
        n += H(a_raw, i)
        nd += H(a_raw, i, meta=False)
    rows.append(("free-position permutation", f"{n}/{len(wm)}",
                 "expect 0"))
    rows.append(("permutation + delete meta", f"{nd}/{len(wm)}",
                 "expect 0"))

    # -- rebinding ---------------------------------------------------
    n = sum(H(wm[i], i, meta=False) for i in range(len(wm)))
    rows.append(("rebinding (metadata deleted)", f"{n}/{len(wm)}",
                 "expect 0"))
    n = 0
    for i in range(len(wm)):
        Y = int(rng.integers(0, len(wm)))
        if Y == i:
            continue
        v = RH.verify_hardened(wm[i], U, mu, e0_band, e0_ppb,
                               meta=metas[i], key=KEY,
                               expected_doc_id=Y)["verdict"]
        n += (v == "PASS")
    rows.append(("index rebind (meta kept)", f"{n}/~{len(wm)}",
                 "expect 0"))

    # -- revoked replay vs manifest ----------------------------------
    revoked = sorted(rng.choice(len(wm), size=50, replace=False).tolist())
    man2 = RH.build_manifest([i for i in doc_ids if i not in set(revoked)],
                             epoch=2, key=KEY)
    n = sum(H(wm[i], i, man=man2) for i in revoked)
    rows.append(("revoked-doc replay", f"{n}/50", "expect 0"))

    # -- flooding replicas + dedup control ---------------------------
    flood = int(rng.integers(0, len(wm)))
    rep_pass = sum(H(wm[flood], flood, man=manifest) for _ in range(20))
    db_docids = doc_ids + [flood] * 20
    ranked = list(range(len(wm), len(wm) + 20)) + [0, 1, 2, 3, 4, 5]
    kept = RH.dedup_topk(ranked, db_docids, 5)
    uniq = len({db_docids[j] for j in kept})
    rows.append(("flooding replicas (byte-genuine)", f"{rep_pass}/20",
                 "pass by design"))
    rows.append(("  ...after doc_id dedup", f"{uniq}/5 unique",
                 "expect 5/5"))

    # -- table -------------------------------------------------------
    print(f"\n{'attack vs hardened verifier':34s} {'PASS':>14s}   note")
    print("-" * 66)
    for name, val, note in rows:
        print(f"{name:34s} {val:>14s}   {note}")
        res[name.strip()] = val

    ok = True
    with open("results/hardened_validation.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("\nSaved results/hardened_validation.json")


if __name__ == "__main__":
    main()
