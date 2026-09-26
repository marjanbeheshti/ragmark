#!/usr/bin/env python3
"""
Dual-Band Security: E_I Band Attack + Storage Transformations
==============================================================
Two experiments:
  1. E_I-only band modification attack: modify [154,384) only,
     keep metadata intact. Tests old (E_A-only) vs new (dual-layer)
     verification.
  2. Storage transformations: float16, bfloat16, int8 (symmetric),
     int8 (asymmetric). Tests false rejection of legitimate vectors.

Usage: python experiments/dualband_security_tests.py
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
    w_raw, meta = R.watermark(v_raw, source_id, U, mu,
                              e1_band, e0_band, e0_ppb,
                              doc_id=source_id, key=key)
    return w_raw, (meta["f_a"], meta["sel"])


def verify_dual(v_raw, U, mu, key, e1_band, e0_band, e0_ppb,
                e1_meta=None, tau=TAU):
    """
    Fixed verification: always check E_I first, then E_A if metadata.
    Returns (source_id, confidence, verdict, details)
    """
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T

    # Step 1: Always check E_I
    sid_i, conf_i, verdict_i = e0_decode(v_pc, key, e0_band, e0_ppb)

    if verdict_i == "REJECTED":
        return None, conf_i, "REJECTED", "E_I rejected"

    # Step 2: If metadata present, also check E_A
    if e1_meta is not None:
        e1_fp, e1_sel = e1_meta
        sid_a, conf_a = decode_e1(v_pc, e1_sel)
        if conf_a < tau:
            return None, conf_a, "REJECTED", "E_A rejected"
        if sid_a != sid_i:
            return None, min(conf_a, conf_i), "REJECTED", "ID mismatch"
        return sid_a, min(conf_a, conf_i), "PASS", "dual-layer"

    # Step 3: No metadata, E_I alone
    return sid_i, conf_i, "PASS", "E_I only"


def verify_ea_only(v_raw, U, mu, e1_meta, tau=TAU):
    """Old (broken) verification: E_A only when metadata present."""
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T
    e1_fp, e1_sel = e1_meta
    sid, conf = decode_e1(v_pc, e1_sel)
    verdict = "PASS" if conf >= tau else "REJECTED"
    return sid, conf, verdict


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(2026)
    results = {"date": datetime.now().isoformat()}

    print("=" * 60)
    print("Dual-Band Security: E_I Band Attack + Storage Tests")
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

    # Watermark all
    print("  Watermarking...")
    wm_raw = []
    e1_metas = []
    for i in range(len(E)):
        w, meta = embed_full(E[i], i % 8, U, mu, KEY,
                             e1_band, e0_band, e0_ppb)
        wm_raw.append(w)
        e1_metas.append(meta)
    wm_raw = np.array(wm_raw)

    # Baseline
    ok_dual = sum(verify_dual(wm_raw[i], U, mu, KEY, e1_band, e0_band,
                              e0_ppb, e1_metas[i])[2] == "PASS"
                  for i in range(len(wm_raw)))
    print(f"  Baseline dual-layer: {ok_dual}/500 PASS")

    # ================================================================
    # TEST 1: E_I-ONLY BAND MODIFICATION ATTACK
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: E_I-Only Band Modification Attack")
    print("=" * 60)
    print("Attack: modify only E_I band [154,384), keep metadata intact")

    for noise_sigma in [0.001, 0.01, 0.05, 0.1]:
        ea_only_pass = 0
        dual_pass = 0
        for i in range(len(wm_raw)):
            # Modify only E_I band in PC space
            v_pc = (np.asarray(wm_raw[i], dtype=mu.dtype) - mu) @ U.T
            v_pc[e0_band[0]:e0_band[1]] += rng.normal(
                0, noise_sigma, e0_band[1] - e0_band[0])
            modified_raw = v_pc @ U + mu

            # Old verification (E_A only) — should PASS (bug)
            _, _, verd_old = verify_ea_only(modified_raw, U, mu,
                                            e1_metas[i])
            ea_only_pass += (verd_old == "PASS")

            # New verification (dual-layer) — should REJECT (fix)
            _, _, verd_new, _ = verify_dual(modified_raw, U, mu, KEY,
                                            e1_band, e0_band, e0_ppb,
                                            e1_metas[i])
            dual_pass += (verd_new == "PASS")

        print(f"\n  sigma={noise_sigma}:")
        print(f"    E_A-only (old):  {ea_only_pass}/500 PASS "
              f"{'*** VULNERABILITY ***' if ea_only_pass > 0 else ''}")
        print(f"    Dual-layer (new): {dual_pass}/500 PASS "
              f"{'(all rejected)' if dual_pass == 0 else ''}")

    # Test with adversarial replacement (not just noise)
    print("\n  --- Adversarial replacement: zero E_I band ---")
    ea_pass_zero = 0
    dual_pass_zero = 0
    for i in range(len(wm_raw)):
        v_pc = (np.asarray(wm_raw[i], dtype=mu.dtype) - mu) @ U.T
        v_pc[e0_band[0]:e0_band[1]] = 0.0  # zero out E_I band
        modified_raw = v_pc @ U + mu
        _, _, v_old = verify_ea_only(modified_raw, U, mu, e1_metas[i])
        ea_pass_zero += (v_old == "PASS")
        _, _, v_new, _ = verify_dual(modified_raw, U, mu, KEY,
                                      e1_band, e0_band, e0_ppb,
                                      e1_metas[i])
        dual_pass_zero += (v_new == "PASS")
    print(f"    E_A-only (old):  {ea_pass_zero}/500 PASS")
    print(f"    Dual-layer (new): {dual_pass_zero}/500 PASS")

    # Test with attacker-chosen values in E_I band
    print("\n  --- Adversarial replacement: random values in E_I band ---")
    ea_pass_rand = 0
    dual_pass_rand = 0
    for i in range(len(wm_raw)):
        v_pc = (np.asarray(wm_raw[i], dtype=mu.dtype) - mu) @ U.T
        v_pc[e0_band[0]:e0_band[1]] = rng.normal(
            0, 0.1, e0_band[1] - e0_band[0])
        modified_raw = v_pc @ U + mu
        _, _, v_old = verify_ea_only(modified_raw, U, mu, e1_metas[i])
        ea_pass_rand += (v_old == "PASS")
        _, _, v_new, _ = verify_dual(modified_raw, U, mu, KEY,
                                      e1_band, e0_band, e0_ppb,
                                      e1_metas[i])
        dual_pass_rand += (v_new == "PASS")
    print(f"    E_A-only (old):  {ea_pass_rand}/500 PASS")
    print(f"    Dual-layer (new): {dual_pass_rand}/500 PASS")

    results["ei_band_attack"] = {
        "ea_only_pass_sigma01": ea_only_pass,
        "dual_pass_sigma01": dual_pass,
        "ea_only_zero": ea_pass_zero,
        "dual_zero": dual_pass_zero,
        "ea_only_rand": ea_pass_rand,
        "dual_rand": dual_pass_rand,
    }

    # ================================================================
    # TEST 2: STORAGE TRANSFORMATIONS
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Storage Transformations (False Rejection Rate)")
    print("=" * 60)

    transforms = {
        "float32 (baseline)": lambda x: x,
        "float16": lambda x: x.astype(np.float16).astype(np.float32),
        "bfloat16": lambda x: (
            np.frombuffer(
                np.array(x, dtype=np.float32).tobytes(),
                dtype=np.uint32
            ).reshape(x.shape).__rshift__(16).__lshift__(16)
            .view(np.float32).copy()
        ),
        "int8 (symmetric)": lambda x: (
            np.round(x * 127.0 / (np.max(np.abs(x)) + 1e-12)
                     ).astype(np.int8).astype(np.float32)
            * (np.max(np.abs(x)) + 1e-12) / 127.0
        ),
    }

    for name, transform in transforms.items():
        ea_pass = ei_pass = dual_pass = 0
        cos_sum = 0.0
        for i in range(len(wm_raw)):
            w_t = transform(wm_raw[i])
            cos_sum += np.dot(wm_raw[i], w_t) / (
                np.linalg.norm(wm_raw[i]) * np.linalg.norm(w_t) + 1e-12)

            # E_A check
            sid_a, conf_a = decode_e1(
                (np.asarray(w_t, dtype=mu.dtype) - mu) @ U.T,
                e1_metas[i][1])
            ea_pass += (conf_a >= TAU and sid_a == i % 8)

            # E_I check
            _, _, verd_i = e0_decode(
                (np.asarray(w_t, dtype=mu.dtype) - mu) @ U.T,
                KEY, e0_band, e0_ppb)
            ei_pass += (verd_i == "PASS")

            # Dual-layer
            _, _, verd_d, _ = verify_dual(w_t, U, mu, KEY,
                                          e1_band, e0_band, e0_ppb,
                                          e1_metas[i])
            dual_pass += (verd_d == "PASS")

        cos_avg = cos_sum / len(wm_raw)
        print(f"\n  {name}:")
        print(f"    CosSim vs original: {cos_avg:.6f}")
        print(f"    E_A self-check: {ea_pass}/500")
        print(f"    E_I self-check: {ei_pass}/500")
        print(f"    Dual-layer:     {dual_pass}/500")

        results[f"storage_{name.split()[0].lower()}"] = {
            "cos": float(cos_avg),
            "ea_pass": ea_pass,
            "ei_pass": ei_pass,
            "dual_pass": dual_pass,
        }

    # ================================================================
    # TEST 3: ROC-STYLE THRESHOLD SWEEP
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Confidence Distributions + Threshold Sweep")
    print("=" * 60)

    # Collect confidences from different populations
    conf_legit = []  # legitimate watermarked vectors
    conf_inject = []  # random injection vectors
    conf_centroid = []  # centroid-averaged vectors
    conf_poison = []  # PoisonedRAG vectors

    # Legitimate
    for i in range(len(wm_raw)):
        _, conf, _ = e0_decode(
            (np.asarray(wm_raw[i], dtype=mu.dtype) - mu) @ U.T,
            KEY, e0_band, e0_ppb)
        conf_legit.append(conf)

    # Random injection
    for _ in range(500):
        v = rng.normal(0, 1, d).astype(np.float32)
        v /= np.linalg.norm(v)
        _, conf, _ = e0_decode(
            (np.asarray(v, dtype=mu.dtype) - mu) @ U.T,
            KEY, e0_band, e0_ppb)
        conf_inject.append(conf)

    # Centroids
    for _ in range(200):
        idx = rng.choice(len(wm_raw), 100, replace=False)
        c = wm_raw[idx].mean(axis=0)
        c /= np.linalg.norm(c) + 1e-12
        _, conf, _ = e0_decode(
            (np.asarray(c, dtype=mu.dtype) - mu) @ U.T,
            KEY, e0_band, e0_ppb)
        conf_centroid.append(conf)

    # PoisonedRAG
    prag_path = os.path.expanduser(
        "~/regmark/PoisonedRAG/results/adv_targeted_results/nq.json")
    if os.path.exists(prag_path):
        prag = json.load(open(prag_path))
        poison_texts = []
        for k in prag:
            poison_texts.extend(prag[k]["adv_texts"])
        Pp = np.asarray(model.encode(poison_texts[:500],
                                      normalize_embeddings=True))
        for i in range(len(Pp)):
            _, conf, _ = e0_decode(
                (np.asarray(Pp[i], dtype=mu.dtype) - mu) @ U.T,
                KEY, e0_band, e0_ppb)
            conf_poison.append(conf)

    print(f"\n  Confidence distributions:")
    print(f"    Legitimate:  mean={np.mean(conf_legit):.3f}  "
          f"min={np.min(conf_legit):.3f}  "
          f"std={np.std(conf_legit):.3f}")
    print(f"    Random:      mean={np.mean(conf_inject):.3f}  "
          f"max={np.max(conf_inject):.3f}  "
          f"std={np.std(conf_inject):.3f}")
    print(f"    Centroid:    mean={np.mean(conf_centroid):.3f}  "
          f"max={np.max(conf_centroid):.3f}  "
          f"std={np.std(conf_centroid):.3f}")
    if conf_poison:
        print(f"    PoisonedRAG: mean={np.mean(conf_poison):.3f}  "
              f"max={np.max(conf_poison):.3f}  "
              f"std={np.std(conf_poison):.3f}")

    # Threshold sweep
    print(f"\n  {'tau':>6s} {'FNR':>8s} {'FPR_rand':>10s} "
          f"{'FPR_cent':>10s} {'FPR_pois':>10s}")
    print("  " + "-" * 46)
    for tau_test in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                     0.80, 0.85, 0.90, 0.95, 1.00]:
        fnr = sum(c < tau_test for c in conf_legit) / len(conf_legit)
        fpr_r = sum(c >= tau_test for c in conf_inject) / len(conf_inject)
        fpr_c = sum(c >= tau_test for c in conf_centroid) / len(conf_centroid)
        fpr_p = (sum(c >= tau_test for c in conf_poison) / len(conf_poison)
                 if conf_poison else 0)
        print(f"  {tau_test:6.2f} {fnr:8.3f} {fpr_r:10.3f} "
              f"{fpr_c:10.3f} {fpr_p:10.3f}")

    results["confidence_distributions"] = {
        "legit_mean": float(np.mean(conf_legit)),
        "legit_min": float(np.min(conf_legit)),
        "inject_mean": float(np.mean(conf_inject)),
        "inject_max": float(np.max(conf_inject)),
        "centroid_mean": float(np.mean(conf_centroid)),
        "centroid_max": float(np.max(conf_centroid)),
    }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    with open("results/security_tests.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved results/security_tests.json")


if __name__ == "__main__":
    main()
