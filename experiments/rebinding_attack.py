#!/usr/bin/env python3
"""
Rebinding via Metadata Deletion (Walkthrough Sec. 6.2)
======================================================
Zero modification of any vector. The attacker takes document X's
legitimate watermarked vector, deletes its metadata blob, and re-inserts
the (byte-identical) vector under doc_id Y. Integrity passes (vector
untouched); no metadata -> PASS (source unknown). The HMAC -- the only
thing binding vector to doc_id -- never runs. Queries matching X now
dereference Y's text.

Note h_v is IRRELEVANT here: the vector is byte-identical, so h_v would
match. Only STRICT MODE (reject vectors without metadata) closes it.
A secondary variant keeps the metadata attached but rebinds the index
row's doc_id -- closed by the doc_id/index binding check.

  (A) Base scheme: delete meta, rebind X->Y. Measure the fraction of
      X-queries whose top-5 now dereferences Y's text, and confirm the
      rebinding entry verifies as PASS (unattributed).
  (B) Strict mode: same attack -> rejected (no metadata).
  (C) Metadata-attached index rebind -> caught by doc_id binding check.

Writes results/rebinding_attack.json
Usage: python experiments/rebinding_attack.py
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
from sklearn.decomposition import PCA

N_DOCS = 500
N_QUERIES = 500
KEY = R.MASTER_KEY
SEED = 2026


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"date": datetime.now().isoformat(),
           "attack": "rebinding via metadata deletion", "seed": SEED}

    print("=" * 62)
    print("Rebinding via Metadata Deletion (Sec. 6.2)")
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
    e1_band, e0_band, e0_ppb = R.compute_bands(d)
    print(f"  d={d}, E_A={e1_band}, E_I={e0_band}, ppb={e0_ppb}")

    print("  Watermarking...")
    wm, metas = [], []
    for i in range(len(E)):
        w_raw, meta = RH.watermark_hardened(E[i], i % 8, U, mu,
                                            e1_band, e0_band, e0_ppb,
                                            doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta)
    wm = np.asarray(wm)

    # queries + ground truth doc for each
    text2idx = {t: i for i, t in enumerate(texts)}
    qa = []
    for r in ds:
        if r["context"] in text2idx:
            qa.append((r["question"], text2idx[r["context"]]))
        if len(qa) >= 3000:
            break
    rng.shuffle(qa)
    qa = qa[:N_QUERIES]
    Q = np.asarray(model.encode([q for q, _ in qa],
                                normalize_embeddings=True))
    gt = np.array([i for _, i in qa])
    s_wm = unit(Q) @ unit(wm).T
    rank_wm = np.argsort(-s_wm, axis=1)

    # =================================================================
    # (A) Base scheme: delete X's meta, rebind vector to Y's doc_id.
    #     The rebound entry keeps X's VECTOR (so X-queries still match
    #     it) but now points at doc_id Y (dereferences Y's text).
    # =================================================================
    print("\n[A] Base scheme: delete metadata, rebind X -> Y")
    misdirected = 0
    entry_pass = 0
    trials = 0
    for qi in range(len(Q)):
        X = int(gt[qi])                       # doc X is the GT for query qi
        # pick a target Y != X to smuggle in
        Y = int(rng.integers(0, len(wm)))
        if Y == X:
            continue
        trials += 1
        # attacker inserts X's vector, metadata stripped, labeled doc_id Y
        entry_vec = wm[X]                      # byte-identical, untouched
        v = R.verify(entry_vec, U, mu, e0_band, e0_ppb,
                     meta=None, key=KEY)       # no metadata path
        entry_pass += (v["verdict"] == "PASS")
        # does this entry land in top-5 for query qi (matching X)?
        s = float(unit(Q[qi]) @ unit(entry_vec))
        col = s_wm[qi].copy(); col[X] = s      # entry occupies X's slot
        rank = int(np.sum(col > s))
        # if it is retrieved, the dereferenced text is Y's, not X's
        misdirected += (rank < 5)
    print(f"  Rebound entry PASS (unattributed): {entry_pass}/{trials}")
    print(f"  X-query top-5 now dereferences Y:  {misdirected}/{trials} "
          f"({100*misdirected/max(trials,1):.2f}%)")
    res["base_entry_pass"] = f"{entry_pass}/{trials}"
    res["base_misdirected"] = f"{misdirected}/{trials}"
    res["base_misdirected_pct"] = float(100*misdirected/max(trials, 1))

    # =================================================================
    # (B) Strict mode: same rebound entry (no metadata) -> reject
    # =================================================================
    print("\n[B] Strict mode: rebound entry (no metadata)")
    strict_pass = 0
    for qi in range(min(len(Q), 500)):
        X = int(gt[qi])
        v = RH.verify_hardened(wm[X], U, mu, e0_band, e0_ppb,
                               meta=None, key=KEY, strict=True)
        strict_pass += (v["verdict"] == "PASS")
    print(f"  Rebound entry PASS under strict:   {strict_pass}/"
          f"{min(len(Q),500)}  (expect 0)")
    res["strict_pass"] = f"{strict_pass}/{min(len(Q),500)}"

    # =================================================================
    # (C) Metadata-attached index rebind: keep X's (vector, meta) but
    #     the index row claims doc_id Y. Caught by doc_id binding check.
    # =================================================================
    print("\n[C] Metadata attached, index row rebound X -> Y")
    bind_pass = 0
    trials_c = 0
    for i in range(min(len(wm), 500)):
        X = i
        Y = int(rng.integers(0, len(wm)))
        if Y == X:
            continue
        trials_c += 1
        # index claims doc_id Y; signed meta says doc_id X
        v = RH.verify_hardened(wm[X], U, mu, e0_band, e0_ppb,
                               meta=metas[X], key=KEY,
                               expected_doc_id=Y)
        bind_pass += (v["verdict"] == "PASS")
    print(f"  Index-rebind PASS w/ binding check: {bind_pass}/{trials_c}  "
          f"(expect 0)")
    res["index_rebind_pass"] = f"{bind_pass}/{trials_c}"

    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  base: {res['base_misdirected_pct']:.1f}% of X-queries "
          f"misdirected to Y")
    print(f"  strict mode closes it: {strict_pass}/{min(len(Q),500)} pass")
    print(f"  doc_id binding closes index-rebind: {bind_pass}/{trials_c} "
          f"pass")
    print("=" * 62)
    with open("results/rebinding_attack.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/rebinding_attack.json")


if __name__ == "__main__":
    main()
