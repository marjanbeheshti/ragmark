#!/usr/bin/env python3
"""
Duplicate Flooding (Walkthrough Sec. 6.3)
=========================================
The attacker copies a legitimate (vector, metadata) tuple wholesale and
inserts N byte-identical replicas. Every replica passes fully attributed
(the replay defenses only fire when a blob moves to a DIFFERENT vector).
Top-k fills with duplicates of one document; context diversity collapses.
Retrieval-denial, not poisoning. h_v and strict mode are powerless (every
copy is byte-genuine). Closed by doc_id deduplication at the gate.

  (A) Base scheme: insert N replicas of a chosen "loud" document; every
      replica PASSes; measure unique docs in top-5 and GT Recall@5.
  (B) With gate-side dedup: replicas collapse to one; diversity restored.

Writes results/flooding_attack.json
Usage: python experiments/flooding_attack.py
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
N_REPLICAS = [5, 20, 100]


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"date": datetime.now().isoformat(),
           "attack": "duplicate flooding", "seed": SEED,
           "replica_counts": N_REPLICAS}

    print("=" * 62)
    print("Duplicate Flooding (Sec. 6.3)")
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
    wm, metas, doc_ids = [], [], []
    for i in range(len(E)):
        w_raw, meta = RH.watermark_hardened(E[i], i % 8, U, mu,
                                            e1_band, e0_band, e0_ppb,
                                            doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta); doc_ids.append(i)
    wm = np.asarray(wm)

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

    def topk_stats(db_vecs, db_docids, k=5):
        """Return (mean unique docs in top-k, GT recall@k) over queries."""
        s = unit(Q) @ unit(db_vecs).T
        rank = np.argsort(-s, axis=1)
        uniq, rec = [], []
        for qi in range(len(Q)):
            top = rank[qi, :k]
            uniq.append(len({db_docids[j] for j in top}))
            rec.append(any(db_docids[j] == gt[qi] for j in top))
        return float(np.mean(uniq)), float(np.mean(rec))

    def topk_stats_dedup(db_vecs, db_docids, k=5):
        """Same, but apply gate-side doc_id dedup before taking top-k."""
        s = unit(Q) @ unit(db_vecs).T
        rank = np.argsort(-s, axis=1)
        uniq, rec = [], []
        for qi in range(len(Q)):
            kept = RH.dedup_topk(rank[qi], db_docids, k)
            uniq.append(len({db_docids[j] for j in kept}))
            rec.append(any(db_docids[j] == gt[qi] for j in kept))
        return float(np.mean(uniq)), float(np.mean(rec))

    # clean baseline
    u0, r0 = topk_stats(wm, doc_ids)
    print(f"\n  Clean baseline: unique docs in top-5={u0:.3f}, "
          f"GT Recall@5={r0:.3f}")
    res["clean_unique_top5"] = u0
    res["clean_recall5"] = r0

    # =================================================================
    # [A] GLOBAL single-doc flooding (honest baseline).
    # Flood the globally-loudest document and average over ALL queries.
    # Expected weak: identical copies only compete for queries where
    # that one doc is already relevant, so the aggregate barely moves.
    # =================================================================
    print("\n[A] Global single-doc flooding (loudest doc, all queries)")
    mean_sim = (unit(Q) @ unit(wm).T).mean(axis=0)
    flood_doc = int(np.argmax(mean_sim))
    print(f"  Flooding doc_id={flood_doc} (highest mean query sim)")
    res["flood_doc"] = flood_doc
    for N in N_REPLICAS:
        rep_vecs = np.repeat(wm[flood_doc][None, :], N, axis=0)
        db_vecs = np.vstack([wm, rep_vecs])
        db_docids = doc_ids + [flood_doc] * N
        vpass = sum(R.verify(rep_vecs[j], U, mu, e0_band, e0_ppb,
                             meta=metas[flood_doc], key=KEY)["verdict"]
                    == "PASS" for j in range(N))
        u1, r1 = topk_stats(db_vecs, db_docids)
        u2, r2 = topk_stats_dedup(db_vecs, db_docids)
        print(f"  N={N:3d}: replicas PASS {vpass}/{N}  |  "
              f"no-dedup unique={u1:.3f} recall={r1:.3f}  ->  "
              f"dedup unique={u2:.3f} recall={r2:.3f}")
        res[f"global_N{N}_replicas_pass"] = f"{vpass}/{N}"
        res[f"global_N{N}_nodedup_unique"] = u1
        res[f"global_N{N}_nodedup_recall5"] = r1
        res[f"global_N{N}_dedup_unique"] = u2
        res[f"global_N{N}_dedup_recall5"] = r2

    # =================================================================
    # [B] TARGETED flooding (the real threat).
    # For each query, flood THAT query's rank-1 document, then measure
    # THAT query's top-5. Identical copies occupy the top slots, so the
    # 5 distinct results collapse toward a single document; and where
    # the GT doc was not rank-1, it is pushed out of the top-5.
    # doc_id dedup restores both.
    # =================================================================
    print("\n[B] Targeted flooding (flood each query's rank-1 doc)")
    s_wm = unit(Q) @ unit(wm).T
    rank_wm = np.argsort(-s_wm, axis=1)
    for N in N_REPLICAS:
        uniq_nd, uniq_dd, rec_nd, rec_dd = [], [], [], []
        vpass = vtot = 0
        for qi in range(len(Q)):
            D = int(rank_wm[qi, 0])                 # this query's top doc
            # augmented DB = originals + N copies of D (shared doc_id D)
            rep = np.repeat(wm[D][None, :], N, axis=0)
            db_vecs = np.vstack([wm, rep])
            db_docids = doc_ids + [D] * N
            s = unit(Q[qi]) @ unit(db_vecs).T
            order = np.argsort(-s)
            # no dedup: first 5 entries
            top = order[:5]
            uniq_nd.append(len({db_docids[j] for j in top}))
            rec_nd.append(any(db_docids[j] == gt[qi] for j in top))
            # dedup: first 5 DISTINCT doc_ids
            kept = RH.dedup_topk(order, db_docids, 5)
            uniq_dd.append(len({db_docids[j] for j in kept}))
            rec_dd.append(any(db_docids[j] == gt[qi] for j in kept))
            if qi < 50:                             # spot-check replicas pass
                vtot += N
                vpass += sum(R.verify(rep[j], U, mu, e0_band, e0_ppb,
                             meta=metas[D], key=KEY)["verdict"] == "PASS"
                             for j in range(N))
        print(f"  N={N:3d}: replicas PASS {vpass}/{vtot} (spot)  |  "
              f"no-dedup unique={np.mean(uniq_nd):.3f} "
              f"recall={np.mean(rec_nd):.3f}  ->  "
              f"dedup unique={np.mean(uniq_dd):.3f} "
              f"recall={np.mean(rec_dd):.3f}")
        res[f"targeted_N{N}_nodedup_unique"] = float(np.mean(uniq_nd))
        res[f"targeted_N{N}_nodedup_recall5"] = float(np.mean(rec_nd))
        res[f"targeted_N{N}_dedup_unique"] = float(np.mean(uniq_dd))
        res[f"targeted_N{N}_dedup_recall5"] = float(np.mean(rec_dd))

    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  clean: unique top-5={u0:.2f}, recall@5={r0:.2f}")
    print("  [A] global single-doc flooding (all queries): negligible")
    for N in N_REPLICAS:
        print(f"      N={N:3d}: unique "
              f"{res[f'global_N{N}_nodedup_unique']:.2f} "
              f"recall {res[f'global_N{N}_nodedup_recall5']:.2f}")
    print("  [B] targeted flooding (per-query rank-1 doc):")
    for N in N_REPLICAS:
        print(f"      N={N:3d}: no-dedup unique="
              f"{res[f'targeted_N{N}_nodedup_unique']:.2f} "
              f"recall={res[f'targeted_N{N}_nodedup_recall5']:.2f}  ->  "
              f"dedup unique={res[f'targeted_N{N}_dedup_unique']:.2f} "
              f"recall={res[f'targeted_N{N}_dedup_recall5']:.2f}")
    print("=" * 62)
    with open("results/flooding_attack.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/flooding_attack.json")


if __name__ == "__main__":
    main()
