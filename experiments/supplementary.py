#!/usr/bin/env python3
"""
Dual-Band Supplementary Experiments
=====================================
Missing experiments for the dual-band + greedy + full-vector fp design:
  1. KS statistical invisibility (E1 band, E0' band, full vector)
  2. Retrieval utility impact (Spearman, NDCG, rank shift)
  3. Scale GT recall (10k docs)

Usage: python experiments/dualband_supplementary.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import ragmark as R
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datasets import load_dataset


# =====================================================================
# Configuration
# =====================================================================
N_BITS = 8
PPB_E1 = 3
DELTA = 0.02
TAU = 0.85
KEY = R.MASTER_KEY


def compute_bands(d):
    return R.compute_bands(d)


def greedy_select_e1(v_pc, source_id, fingerprint, key, band, **kw):
    """Shim -> ragmark.embed_e1. Returns (w, selection)."""
    return R.embed_e1(v_pc, source_id, fingerprint, key, band)


def full_fp(v_pc):
    return R.full_fingerprint(v_pc)


def e0_embed(v_pc, source_id, key, band, ppb, n_bits=None):
    return R.embed_e0(v_pc, source_id, key, band, ppb)


def embed_full(v_raw, source_id, U, mu, key, e1_band, e0_band, e0_ppb):
    """Shim -> ragmark.watermark; returns bare w_raw (caller arrays it)."""
    w_raw, _ = R.watermark(v_raw, source_id, U, mu,
                           e1_band, e0_band, e0_ppb,
                           doc_id=source_id, key=key)
    return w_raw


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(2026)
    results = {"date": datetime.now().isoformat(),
               "design": "dual-band + greedy + full-vec-fp"}

    print("=" * 60)
    print("Dual-Band Supplementary Experiments")
    print("=" * 60)

    # Setup
    print("\n[SETUP] Loading corpus (500 docs)...")
    ds = load_dataset("rajpurkar/squad", split="train")
    seen, texts = set(), []
    for r in ds:
        if r["context"] not in seen and len(texts) < 500:
            seen.add(r["context"])
            texts.append(r["context"])
    model = SentenceTransformer("all-mpnet-base-v2")
    E = np.asarray(model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=True))
    d = E.shape[1]
    pca = PCA(n_components=min(d, len(E))).fit(E)
    U, mu = pca.components_, pca.mean_
    e1_band, e0_band, e0_ppb = compute_bands(d)
    print(f"  d={d}, E1={e1_band}, E0'={e0_band}")

    # Watermark
    print("  Watermarking...")
    wm = np.array([embed_full(E[i], i % 8, U, mu, KEY,
                               e1_band, e0_band, e0_ppb)
                    for i in range(len(E))])
    cos = float(np.mean(np.sum(E * wm, axis=1) /
        (np.linalg.norm(E, axis=1) * np.linalg.norm(wm, axis=1)
         + 1e-12)))
    print(f"  CosSim: {cos:.4f}")

    # PC-space representations
    clean_pc = np.array([(np.asarray(E[i], dtype=mu.dtype) - mu) @ U.T
                         for i in range(len(E))])
    wm_pc = np.array([(np.asarray(wm[i], dtype=mu.dtype) - mu) @ U.T
                       for i in range(len(wm))])

    # ================================================================
    # 1. KS STATISTICAL INVISIBILITY
    # ================================================================
    print("\n[1/3] KS Statistical Invisibility")

    regions = [
        ("E1 band", e1_band[0], e1_band[1]),
        ("E0' band", e0_band[0], e0_band[1]),
        ("Pre-E1 [0,E1_lo)", 0, e1_band[0]),
        ("Post-E0' [E0_hi,d)", e0_band[1], min(d, len(E))),
        ("Full vector", 0, min(d, len(E))),
    ]

    for label, lo, hi in regions:
        if hi <= lo:
            print(f"  {label}: empty region, skipped")
            continue
        c = clean_pc[:, lo:hi].flatten()
        w = wm_pc[:, lo:hi].flatten()
        ks_stat, ks_p = scipy_stats.ks_2samp(c, w)
        print(f"  {label:30s}  KS stat={ks_stat:.6f}  p={ks_p:.4f}")
        results[f"ks_{label.replace(' ', '_').lower()}"] = {
            "stat": float(ks_stat), "p": float(ks_p)}

    # Per-PC KS in E1 band
    e1_reject = 0
    for pc in range(e1_band[0], e1_band[1]):
        _, p = scipy_stats.ks_2samp(clean_pc[:, pc], wm_pc[:, pc])
        if p < 0.05:
            e1_reject += 1
    e1_width = e1_band[1] - e1_band[0]
    print(f"\n  E1 per-PC rejections: {e1_reject}/{e1_width} "
          f"(expected ~{e1_width * 0.05:.0f} by chance)")

    # Per-PC KS in E0' band
    e0_reject = 0
    for pc in range(e0_band[0], e0_band[1]):
        _, p = scipy_stats.ks_2samp(clean_pc[:, pc], wm_pc[:, pc])
        if p < 0.05:
            e0_reject += 1
    e0_width = e0_band[1] - e0_band[0]
    print(f"  E0' per-PC rejections: {e0_reject}/{e0_width} "
          f"(expected ~{e0_width * 0.05:.0f} by chance)")

    # Variance ratios
    for label, lo, hi in [("E1", e1_band[0], e1_band[1]),
                          ("E0'", e0_band[0], e0_band[1])]:
        cv = np.var(clean_pc[:, lo:hi], axis=0)
        wv = np.var(wm_pc[:, lo:hi], axis=0)
        ratio = wv / (cv + 1e-12)
        print(f"  {label} variance ratio: mean={np.mean(ratio):.4f} "
              f"std={np.std(ratio):.4f}")
        results[f"var_ratio_{label.lower()}"] = {
            "mean": float(np.mean(ratio)),
            "std": float(np.std(ratio))}

    results["ks_e1_per_pc_reject"] = e1_reject
    results["ks_e0_per_pc_reject"] = e0_reject

    # Fraction of E1 pairs that were forced (create new values)
    forced = 0
    total = 0
    for i in range(len(E)):
        v = (np.asarray(E[i], dtype=mu.dtype) - mu) @ U.T
        e1_fp = R.content_fingerprint(E[i])
        _, sel_meta = greedy_select_e1(v, i % 8, e1_fp, KEY, e1_band)
        for b in range(N_BITS):
            bit_val = (i % 8 >> b) & 1
            for a, b_ in sel_meta[b]:
                total += 1
                if bit_val == 1 and v[a] < v[b_]:
                    forced += 1
                elif bit_val == 0 and v[a] > v[b_]:
                    forced += 1
    print(f"\n  E1 pairs forced: {forced}/{total} "
          f"({100*forced/total:.1f}%)")
    results["e1_pairs_forced_pct"] = 100 * forced / total

    # ================================================================
    # 2. RETRIEVAL UTILITY IMPACT
    # ================================================================
    print("\n[2/3] Retrieval Utility Impact")

    text2idx = {t: i for i, t in enumerate(texts)}
    qa = []
    for r in ds:
        if r["context"] in text2idx:
            qa.append((r["question"], text2idx[r["context"]]))
        if len(qa) >= 2500:
            break
    rng.shuffle(qa)
    qa = qa[:500]
    Q = np.asarray(model.encode([q for q, _ in qa],
                                normalize_embeddings=True))
    gt_idx = np.array([i for _, i in qa])

    # Similarity matrices
    def cosine_sims(db, queries):
        dbn = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-12)
        qn = queries / (np.linalg.norm(queries, axis=1,
                                        keepdims=True) + 1e-12)
        return qn @ dbn.T

    sims_clean = cosine_sims(E, Q)
    sims_wm = cosine_sims(wm, Q)
    rank_clean = np.argsort(-sims_clean, axis=1)
    rank_wm = np.argsort(-sims_wm, axis=1)

    # Spearman
    rhos = []
    for i in range(len(Q)):
        r, _ = scipy_stats.spearmanr(sims_clean[i], sims_wm[i])
        rhos.append(r)
    spearman = np.mean(rhos)
    print(f"  Spearman rank correlation: {spearman:.4f}")
    results["spearman"] = float(spearman)

    # Top-k overlap
    for k in [1, 5, 10]:
        overlaps = []
        for i in range(len(Q)):
            sa = set(rank_clean[i, :k])
            sb = set(rank_wm[i, :k])
            overlaps.append(len(sa & sb) / len(sa | sb))
        olap = np.mean(overlaps)
        print(f"  Top-{k} overlap: {olap:.3f}")
        results[f"top{k}_overlap"] = float(olap)

    # GT Recall@k
    for k in [1, 5, 10]:
        rc = np.mean([gt_idx[i] in rank_clean[i, :k]
                       for i in range(len(gt_idx))])
        rw = np.mean([gt_idx[i] in rank_wm[i, :k]
                       for i in range(len(gt_idx))])
        print(f"  GT Recall@{k}: clean={rc:.3f}  wm={rw:.3f}  "
              f"diff={rw-rc:+.3f}")
        results[f"recall_{k}_clean"] = float(rc)
        results[f"recall_{k}_wm"] = float(rw)

    # MRR@10
    def mrr(rankings, gt, k=10):
        rrs = []
        for i in range(len(gt)):
            pos = np.where(rankings[i, :k] == gt[i])[0]
            rrs.append(1.0 / (pos[0] + 1) if len(pos) > 0 else 0.0)
        return np.mean(rrs)

    mrr_c = mrr(rank_clean, gt_idx)
    mrr_w = mrr(rank_wm, gt_idx)
    print(f"  MRR@10: clean={mrr_c:.4f}  wm={mrr_w:.4f}  "
          f"diff={mrr_w-mrr_c:+.4f}")
    results["mrr_clean"] = float(mrr_c)
    results["mrr_wm"] = float(mrr_w)

    # NDCG@5
    def ndcg(rankings, gt, k=5):
        vals = []
        for i in range(len(gt)):
            dcg = sum(1.0 / np.log2(j + 2)
                      for j in range(k) if rankings[i, j] == gt[i])
            vals.append(dcg / (1.0 / np.log2(2)))
        return np.mean(vals)

    ndcg_c = ndcg(rank_clean, gt_idx)
    ndcg_w = ndcg(rank_wm, gt_idx)
    print(f"  NDCG@5: clean={ndcg_c:.4f}  wm={ndcg_w:.4f}  "
          f"diff={ndcg_w-ndcg_c:+.4f}")
    results["ndcg5_clean"] = float(ndcg_c)
    results["ndcg5_wm"] = float(ndcg_w)

    # Rank shift
    gt_rank_c = []
    gt_rank_w = []
    for i in range(len(gt_idx)):
        pc = np.where(rank_clean[i] == gt_idx[i])[0]
        pw = np.where(rank_wm[i] == gt_idx[i])[0]
        gt_rank_c.append(pc[0] if len(pc) > 0 else 999)
        gt_rank_w.append(pw[0] if len(pw) > 0 else 999)
    shift = np.array(gt_rank_w, dtype=float) - np.array(gt_rank_c,
                                                          dtype=float)
    unchanged = np.sum(np.abs(shift) == 0)
    big_shift = np.sum(np.abs(shift) >= 5)
    print(f"  Rank shift: mean={np.mean(shift):+.2f}  "
          f"std={np.std(shift):.2f}  max={np.max(np.abs(shift)):.0f}")
    print(f"  Docs unchanged: {unchanged}/{len(shift)} "
          f"({100*unchanged/len(shift):.1f}%)")
    print(f"  Docs shifted >=5: {big_shift}/{len(shift)} "
          f"({100*big_shift/len(shift):.1f}%)")
    results["rank_shift_mean"] = float(np.mean(shift))
    results["rank_shift_std"] = float(np.std(shift))
    results["docs_unchanged_pct"] = float(100 * unchanged / len(shift))

    # ================================================================
    # 3. SCALE GT RECALL (10k)
    # ================================================================
    print("\n[3/3] Scale GT Recall (10k docs)")

    texts10 = []
    seen10 = set()
    for r in ds:
        if r["context"] not in seen10 and len(texts10) < 10000:
            seen10.add(r["context"])
            texts10.append(r["context"])

    doc_set10 = set(texts10)
    qa10 = []
    for r in ds:
        if r["context"] in doc_set10:
            qa10.append((r["question"], r["context"]))
        if len(qa10) >= 2500:
            break
    rng.shuffle(qa10)
    qa10 = qa10[:500]

    E10 = np.asarray(model.encode(texts10, normalize_embeddings=True,
                                   batch_size=128, show_progress_bar=True))
    Q10 = np.asarray(model.encode([q for q, _ in qa10],
                                   normalize_embeddings=True))
    gt10 = [c for _, c in qa10]

    pca10 = PCA(n_components=min(d, len(E10))).fit(E10)
    U10, mu10 = pca10.components_, pca10.mean_
    e1b10, e0b10, e0p10 = compute_bands(d)

    wm10 = np.array([embed_full(E10[i], i % 8, U10, mu10, KEY,
                                 e1b10, e0b10, e0p10)
                      for i in range(len(E10))])

    cos10 = float(np.mean(np.sum(E10 * wm10, axis=1) /
        (np.linalg.norm(E10, axis=1) * np.linalg.norm(wm10, axis=1)
         + 1e-12)))

    def gt_recall_scale(db, doc_texts, qv, gt_docs, k=5):
        dbn = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-12)
        qn = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
        top = np.argsort(-(qn @ dbn.T), axis=1)[:, :k]
        return sum(g in {doc_texts[j] for j in top[qi]}
                   for qi, g in enumerate(gt_docs)) / len(gt_docs)

    rc10 = gt_recall_scale(E10, texts10, Q10, gt10)
    rw10 = gt_recall_scale(wm10, texts10, Q10, gt10)
    print(f"  CosSim: {cos10:.4f}")
    print(f"  GT recall clean: {rc10*100:.1f}%")
    print(f"  GT recall watermarked: {rw10*100:.1f}%")
    print(f"  Gap: {(rw10-rc10)*100:+.1f} pts")
    results["scale_cos"] = cos10
    results["scale_recall_clean"] = rc10
    results["scale_recall_wm"] = rw10

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    with open("results/supplementary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved results/supplementary.json")


if __name__ == "__main__":
    main()
