#!/usr/bin/env python3
"""
Scale: RagMark at 100k Documents (Walkthrough Sec. 10)
======================================================
The paper's largest run is 10k SQuAD docs; reviewers will ask how the
scheme behaves at production corpus sizes. This script measures, on a
100k-passage MS MARCO corpus (BeIR/msmarco):

  (A) Cost at scale: embed/verify wall time per vector; PCA fit time.
  (B) Integrity at scale: self-check pass rate over ALL 100k vectors
      (the false-reject trend: 499/500 -> 9991/10k -> ?/100k).
  (C) Utility at scale: GT recall@5 clean vs watermarked on 1,000
      MS MARCO dev queries whose GT passages are in the corpus.
  (D) Black-Hole at scale: 1,000 k-means centroids injected into the
      100k index -- rejection rate.
  (E) Storage sweep at scale: float32 full-corpus; float16 / bfloat16 /
      int8 on a 5k sample (decode cost, not information, limits this).

Embeddings are cached to cache/scale_emb_{n}.npy so reruns skip the
GPU-heavy step.

Usage:
    python experiments/scale_100k.py
    python experiments/scale_100k.py --n-docs 100000 --n-queries 1000

Writes results/scale_100k.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import ragmark as R
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from datasets import load_dataset

KEY = R.MASTER_KEY


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def selfcheck_rate(vecs, U, mu, e0_band, e0_ppb, label=""):
    ok = 0
    t0 = time.perf_counter()
    for i in range(len(vecs)):
        v_pc = (np.asarray(vecs[i], dtype=mu.dtype) - mu) @ U.T
        ok += (R.decode_e0(v_pc, KEY, e0_band, e0_ppb)[2] == "PASS")
        if label and (i + 1) % 20000 == 0:
            print(f"    {label} {i+1}/{len(vecs)}: {ok} pass")
    ms = 1000 * (time.perf_counter() - t0) / len(vecs)
    return ok, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=100_000)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--storage-sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    res = {"date": datetime.now().isoformat(),
           "test": f"scale run, {args.n_docs} MS MARCO passages",
           "n_docs": args.n_docs}

    print("=" * 66)
    print(f"Scale: RagMark at {args.n_docs:,} documents (Sec. 10)")
    print("=" * 66)

    # ---- [1] dev queries + qrels: GT passages that must be in-corpus
    print("\n[1/7] MS MARCO dev queries + qrels...")
    qrels_ds = load_dataset("BeIR/msmarco-qrels", split="validation")
    qrels = {}
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(
                str(r["corpus-id"]))
    qids = sorted(qrels.keys())
    rng.shuffle(qids)
    qids = qids[:args.n_queries]
    wanted = set()
    for q in qids:
        wanted |= qrels[q]
    queries_ds = load_dataset("BeIR/msmarco", "queries", split="queries")
    qtext = {str(r["_id"]): r["text"] for r in queries_ds
             if str(r["_id"]) in set(qids)}
    qids = [q for q in qids if q in qtext]
    print(f"  {len(qids)} queries, {len(wanted)} GT passages to collect")

    # ---- [2] corpus: GT passages + filler to n_docs (streaming) ----
    print("[2/7] Streaming BeIR/msmarco corpus (one pass)...")
    n_filler = args.n_docs - len(wanted)
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus",
                          streaming=True)
    gt_docs, filler, seen_f = {}, [], 0
    for r in corpus:
        did = str(r["_id"])
        text = (r.get("title", "") + " " + r.get("text", "")).strip()
        if did in wanted:
            gt_docs[did] = text
        else:
            seen_f += 1
            if len(filler) < n_filler:
                filler.append(text)
            else:
                j = int(rng.integers(0, seen_f))
                if j < n_filler:
                    filler[j] = text
        if len(gt_docs) == len(wanted) and seen_f >= 3 * n_filler:
            break
    texts, gt_rows = [], {}
    for did, t in gt_docs.items():
        gt_rows[did] = len(texts); texts.append(t)
    texts.extend(filler)
    qids = [q for q in qids if any(d in gt_rows for d in qrels[q])]
    print(f"  Corpus: {len(texts):,} ({len(gt_docs)} GT + "
          f"{len(filler):,} filler); usable queries: {len(qids)}")
    res["corpus_size"] = len(texts)
    res["n_queries_used"] = len(qids)

    # ---- [3] embed (cached) ----
    cache = f"cache/scale_emb_{len(texts)}.npy"
    model = SentenceTransformer("all-mpnet-base-v2")
    if os.path.exists(cache):
        print(f"[3/7] Loading cached embeddings: {cache}")
        E = np.load(cache)
    else:
        print(f"[3/7] Embedding {len(texts):,} passages (cached after)...")
        E = np.asarray(model.encode(texts, normalize_embeddings=True,
                                    batch_size=256,
                                    show_progress_bar=True),
                       dtype=np.float32)
        np.save(cache, E)
    d = E.shape[1]

    # ---- [4] PCA + watermark, timed ----
    print("[4/7] PCA fit + watermark (timed)...")
    t0 = time.perf_counter()
    pca = PCA(n_components=d).fit(E)
    pca_s = time.perf_counter() - t0
    U, mu = pca.components_, pca.mean_
    e1_band, e0_band, e0_ppb = R.compute_bands(d)
    print(f"  PCA fit: {pca_s:.1f}s  bands E_A={e1_band} "
          f"E_I={e0_band} ppb={e0_ppb}")

    t0 = time.perf_counter()
    wm = np.empty_like(E, dtype=np.float64)
    for i in range(len(E)):
        wm[i] = R.watermark(E[i], i % 8, U, mu, e1_band, e0_band,
                            e0_ppb, doc_id=i, key=KEY)[0]
        if (i + 1) % 20000 == 0:
            print(f"    watermarked {i+1:,}/{len(E):,}")
    embed_ms = 1000 * (time.perf_counter() - t0) / len(E)
    cos = float(np.mean(np.sum(E * wm, axis=1) /
                (np.linalg.norm(E, axis=1) *
                 np.linalg.norm(wm, axis=1) + 1e-12)))
    print(f"  Embed: {embed_ms:.3f} ms/vec  CosSim={cos:.4f}")
    res["pca_fit_s"] = pca_s
    res["embed_ms_per_vec"] = embed_ms
    res["cossim"] = cos

    # ---- [5] self-check over ALL vectors (false-reject at scale) ----
    print("[5/7] Self-check over the full corpus...")
    ok, verify_ms = selfcheck_rate(wm, U, mu, e0_band, e0_ppb,
                                   label="self-check")
    print(f"  Self-check: {ok:,}/{len(wm):,} "
          f"({100*ok/len(wm):.3f}%)  verify {verify_ms:.3f} ms/vec")
    print(f"  Trend: 499/500 -> 9991/10k -> {ok}/{len(wm)}")
    res["self_check"] = f"{ok}/{len(wm)}"
    res["verify_ms_per_vec"] = verify_ms

    # ---- [6] utility + Black-Hole at scale ----
    print("[6/7] Recall@5 clean vs watermarked; Black-Hole at scale...")
    Q = np.asarray(model.encode([qtext[q] for q in qids],
                                normalize_embeddings=True))
    En, Wn = unit(E), unit(wm)
    rec_clean = rec_wm = 0
    for qi, q in enumerate(qids):
        gt = {gt_rows[d_] for d_ in qrels[q] if d_ in gt_rows}
        qn = Q[qi]
        rec_clean += bool(gt & set(int(j) for j in
                                   np.argsort(-(En @ qn))[:5]))
        rec_wm += bool(gt & set(int(j) for j in
                                np.argsort(-(Wn @ qn))[:5]))
    nq = len(qids)
    print(f"  Recall@5: clean {rec_clean}/{nq} "
          f"({100*rec_clean/nq:.1f}%)  "
          f"watermarked {rec_wm}/{nq} ({100*rec_wm/nq:.1f}%)  "
          f"delta {100*(rec_wm-rec_clean)/nq:+.1f} pts")
    res["recall_clean"] = f"{rec_clean}/{nq}"
    res["recall_wm"] = f"{rec_wm}/{nq}"

    km = MiniBatchKMeans(n_clusters=1000, random_state=42,
                         batch_size=4096, n_init=3).fit(
        wm.astype(np.float32))
    cents = unit(km.cluster_centers_.astype(np.float64))
    bh_rej = sum(R.decode_e0((cents[i] - mu) @ U.T, KEY, e0_band,
                             e0_ppb)[2] == "REJECTED"
                 for i in range(len(cents)))
    print(f"  Black-Hole centroids rejected: {bh_rej}/1000")
    res["blackhole_rejected"] = f"{bh_rej}/1000"

    # ---- [7] storage sweep ----
    print(f"[7/7] Storage sweep (float32 full; others on "
          f"{args.storage_sample} sample)...")
    idx = rng.choice(len(wm), size=args.storage_sample, replace=False)
    sample = wm[idx]

    def to_f16(x):
        return x.astype(np.float16).astype(np.float64)

    def to_bf16(x):
        try:
            import torch
            return torch.tensor(x).to(torch.bfloat16).to(
                torch.float64).numpy()
        except ImportError:
            try:
                import ml_dtypes
                return x.astype(ml_dtypes.bfloat16).astype(np.float64)
            except ImportError:
                return None

    def to_int8(x):
        s = np.max(np.abs(x), axis=1, keepdims=True) / 127.0
        return np.round(x / s).clip(-127, 127) * s

    sweeps = [("float32", sample.astype(np.float32).astype(np.float64)),
              ("float16", to_f16(sample)),
              ("bfloat16", to_bf16(sample)),
              ("int8 (symmetric)", to_int8(sample))]
    res["storage"] = {}
    for name, arr in sweeps:
        if arr is None:
            print(f"  {name:<18s} SKIPPED (no torch/ml_dtypes)")
            continue
        ok_s, _ = selfcheck_rate(arr, U, mu, e0_band, e0_ppb)
        print(f"  {name:<18s} E_I pass {ok_s}/{len(arr)}")
        res["storage"][name] = f"{ok_s}/{len(arr)}"

    print("\n" + "=" * 66)
    print("SUMMARY")
    print(f"  corpus {len(texts):,} | embed {embed_ms:.2f} ms "
          f"| verify {verify_ms:.2f} ms | PCA {pca_s:.0f}s")
    print(f"  self-check {ok:,}/{len(wm):,} | CosSim {cos:.4f}")
    print(f"  recall@5 clean {100*rec_clean/nq:.1f}% vs wm "
          f"{100*rec_wm/nq:.1f}% | Black-Hole {bh_rej}/1000 rejected")
    print("=" * 66)
    with open("results/scale_100k.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/scale_100k.json")


if __name__ == "__main__":
    main()
