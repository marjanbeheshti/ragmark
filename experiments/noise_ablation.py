#!/usr/bin/env python3
"""
E_A Noise Robustness + Oracle Error Bars
============================================
Two quick experiments:
  1. E_A in-band noise sweep: Gaussian noise on [31,154), test
     whether attribution survives. Compare to E_I noise trichotomy.
  2. Oracle hill-climb error bars: 5 seeds x 10 trials.

Usage: python experiments/dualband_ea_noise_ablation.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import ragmark as R
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datasets import load_dataset

N_BITS = 8
PPB_E1 = 3
DELTA = 0.02
TAU = 0.85
KEY = R.MASTER_KEY


def compute_bands(d):
    return R.compute_bands(d)


def greedy_select_e1(v_pc, source_id, fingerprint, key, band, **kw):
    return R.embed_e1(v_pc, source_id, fingerprint, key, band)


def decode_e1(v_pc, selection_meta, **kw):
    return R.decode_e1(v_pc, selection_meta)


def full_fp(v_pc):
    return R.full_fingerprint(v_pc)


def e0_embed(v_pc, source_id, key, band, ppb, n_bits=None):
    return R.embed_e0(v_pc, source_id, key, band, ppb)


def e0_decode(v_pc, key, band, ppb, n_bits=None, tau=R.TAU):
    return R.decode_e0(v_pc, key, band, ppb, tau=tau)


def embed_full(v_raw, source_id, U, mu, key, e1_band, e0_band, e0_ppb):
    """Returns (w_raw, (f_a, sel)); meta[1] = selection for decode_e1."""
    w_raw, meta = R.watermark(v_raw, source_id, U, mu,
                              e1_band, e0_band, e0_ppb,
                              doc_id=source_id, key=key)
    return w_raw, (meta["f_a"], meta["sel"])


def main():
    os.makedirs("results", exist_ok=True)
    results = {"date": datetime.now().isoformat()}

    print("=" * 60)
    print("E_A Noise Robustness + Oracle Error Bars")
    print("=" * 60)

    # Setup
    print("\n[SETUP] Loading corpus (500 docs)...")
    ds = load_dataset("rajpurkar/squad", split="train")
    seen, texts = set(), []
    for r in ds:
        if r["context"] not in seen and len(texts) < 500:
            seen.add(r["context"]); texts.append(r["context"])
    model = SentenceTransformer("all-mpnet-base-v2")
    E = np.asarray(model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=True))
    d = E.shape[1]
    pca = PCA(n_components=min(d, len(E))).fit(E)
    U, mu = pca.components_, pca.mean_
    e1_band, e0_band, e0_ppb = compute_bands(d)
    print(f"  d={d}, E_A={e1_band}, E_I={e0_band}")

    # Watermark
    print("  Watermarking...")
    wm_raw = []
    e1_metas = []
    for i in range(len(E)):
        w, meta = embed_full(E[i], i % 8, U, mu, KEY,
                             e1_band, e0_band, e0_ppb)
        wm_raw.append(w)
        e1_metas.append(meta)
    wm_raw = np.array(wm_raw)

    # Measure natural gaps for context
    print("\n  Measuring E_A natural gaps...")
    all_gaps = []
    for i in range(len(E)):
        v_pc = (np.asarray(wm_raw[i], dtype=mu.dtype) - mu) @ U.T
        e1_fp, e1_sel = e1_metas[i]
        for b in range(N_BITS):
            bit_val = (i % 8 >> b) & 1
            for a, b_ in e1_sel[b]:
                gap = abs(v_pc[a] - v_pc[b_])
                all_gaps.append(gap)
    gaps = np.array(all_gaps)
    print(f"  E_A gap stats: median={np.median(gaps):.4f}  "
          f"mean={np.mean(gaps):.4f}  min={np.min(gaps):.4f}  "
          f"p10={np.percentile(gaps, 10):.4f}")

    # ================================================================
    # TEST 1: E_A IN-BAND NOISE SWEEP
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: E_A In-Band Noise Sweep")
    print("=" * 60)
    print(f"  Adding Gaussian noise to E_A band [{e1_band[0]},{e1_band[1]})")
    print(f"  Checking: (a) E_A decode with stored sel  "
          f"(b) E_I decode with full-vec fp")

    sigmas = [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    print(f"\n  {'sigma':>10s} {'E_A ok':>8s} {'E_A conf':>10s} "
          f"{'E_I ok':>8s} {'E_I detect':>10s} {'Sum':>6s}")
    print("  " + "-" * 56)

    for sigma in sigmas:
        rng = np.random.default_rng(2026)
        ea_ok = 0
        ei_ok = 0
        ei_det = 0
        ea_confs = []
        for i in range(len(wm_raw)):
            # Add noise only to E_A band
            v_pc = (np.asarray(wm_raw[i], dtype=mu.dtype) - mu) @ U.T
            v_pc[e1_band[0]:e1_band[1]] += rng.normal(
                0, sigma, e1_band[1] - e1_band[0])

            # E_A decode (stored selection)
            sid_a, conf_a = decode_e1(v_pc, e1_metas[i][1])
            ea_confs.append(conf_a)
            if sid_a == i % 8 and conf_a >= TAU:
                ea_ok += 1

            # E_I decode (full-vec fp - noise changes fp!)
            sid_i, conf_i, verd_i = e0_decode(v_pc, KEY,
                                               e0_band, e0_ppb)
            if verd_i == "PASS" and sid_i == i % 8:
                ei_ok += 1
            elif verd_i == "REJECTED":
                ei_det += 1

        print(f"  {sigma:10.0e} {ea_ok:>8d} {np.mean(ea_confs):>10.3f} "
              f"{ei_ok:>8d} {ei_det:>10d} {ei_ok + ei_det:>6d}")

        results[f"ea_noise_{sigma}"] = {
            "ea_ok": ea_ok, "ea_conf_mean": float(np.mean(ea_confs)),
            "ei_ok": ei_ok, "ei_detect": ei_det}

    # Side-by-side with E_I noise for comparison
    print(f"\n  --- E_I noise (for comparison) ---")
    print(f"  {'sigma':>10s} {'E_I decode':>10s} {'E_I detect':>10s} "
          f"{'Sum':>6s}")
    print("  " + "-" * 36)
    for sigma in [1e-5, 1e-4, 1e-3, 5e-3]:
        rng = np.random.default_rng(2026)
        ei_dec = ei_det2 = 0
        for i in range(len(wm_raw)):
            v_pc = (np.asarray(wm_raw[i], dtype=mu.dtype) - mu) @ U.T
            v_pc[e0_band[0]:e0_band[1]] += rng.normal(
                0, sigma, e0_band[1] - e0_band[0])
            sid, conf, verd = e0_decode(v_pc, KEY, e0_band, e0_ppb)
            if verd == "PASS" and sid == i % 8:
                ei_dec += 1
            elif verd == "REJECTED":
                ei_det2 += 1
        print(f"  {sigma:10.0e} {ei_dec:>10d} {ei_det2:>10d} "
              f"{ei_dec + ei_det2:>6d}")

    # ================================================================
    # TEST 2: ABLATION ERROR BARS (5 seeds)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Ablation Error Bars (5 seeds)")
    print("=" * 60)

    # 2a: Content-binding ablation across seeds
    # (content-binding ablation lives in validation.py --
    #  single canonical construction, no duplication here)

    # 2b: Oracle ablation across seeds (full-vec fp)
    print("\n  --- Oracle ablation (full-vec fp, 5 seeds × 10 trials) ---")
    oracle_best_confs = []
    for seed in [2026, 2027, 2028, 2029, 2030]:
        rng_o = np.random.default_rng(seed)
        seed_confs = []
        for trial in range(10):
            v = rng_o.normal(0, 1, d).astype(np.float32)
            v /= np.linalg.norm(v)
            v_pc = (v - mu) @ U.T
            best_conf = 0.0
            for q in range(5000):  # 5k queries per trial (faster)
                idx = rng_o.integers(e0_band[0], e0_band[1])
                _old = v_pc[idx]
                v_pc[idx] += rng_o.normal(0, 0.01)
                _, conf, _ = e0_decode(v_pc, KEY, e0_band, e0_ppb)
                if conf > best_conf:
                    best_conf = conf
                else:
                    v_pc[idx] = _old
                if best_conf >= TAU:
                    break
            seed_confs.append(best_conf)
        forged = sum(c >= TAU for c in seed_confs)
        oracle_best_confs.extend(seed_confs)
        print(f"  seed {seed}: {forged}/10 forged  "
              f"best={max(seed_confs):.3f}  "
              f"median={np.median(seed_confs):.3f}")

    print(f"\n  Overall: {sum(c >= TAU for c in oracle_best_confs)}/50 "
          f"forged")
    print(f"  Confidence: median={np.median(oracle_best_confs):.3f}  "
          f"IQR=[{np.percentile(oracle_best_confs, 25):.3f}, "
          f"{np.percentile(oracle_best_confs, 75):.3f}]  "
          f"max={max(oracle_best_confs):.3f}")

    results["oracle_confs"] = [float(c) for c in oracle_best_confs]

    # Summary
    print(f"\n{'='*60}")
    with open("results/noise_ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved results/noise_ablation.json")


if __name__ == "__main__":
    main()
