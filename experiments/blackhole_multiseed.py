#!/usr/bin/env python3
"""
Black-Hole Multiseed Rerun: Table IV reproducibility artifact
=============================================================
Reproduces Test 1 of e2e_attacks.py exactly (same corpus selection,
same PCA fit, same KMeans random_state=42, same perturbation seed=42,
same 50 random vectors from rng 2026, same QA sampling seeds
2026-2030) -- but this time SAVES the per-seed recalls and computes
the paired t-test (Clean A vs. RagMark C) natively, so the paper's
Table IV numbers (90.2 +/- 1.6 / 50.0 +/- 4.1 / 90.0 +/- 1.6,
poison 4.1, p = 0.37) have a verifiable provenance in the repo.

Retrieval + verification only: no LLM, runs in minutes.

Usage:
    python experiments/blackhole_multiseed.py

Writes results/blackhole_multiseed.json
"""
import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import ragmark as R
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datasets import load_dataset

KEY = R.MASTER_KEY
SEEDS = [2026, 2027, 2028, 2029, 2030]


def embed_full(v_raw, source_id, U, mu, key, e1_band, e0_band, e0_ppb):
    w_raw, meta = R.watermark(v_raw, source_id, U, mu,
                              e1_band, e0_band, e0_ppb,
                              doc_id=source_id, key=key)
    return w_raw


def verify_e0(v_raw, U, mu, key, e0_band, e0_ppb):
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T
    return R.decode_e0(v_pc, key, e0_band, e0_ppb)


def main():
    os.makedirs("results", exist_ok=True)
    res = {"date": datetime.now().isoformat(),
           "test": "Black-Hole multiseed rerun (Table IV provenance)",
           "seeds": SEEDS}

    print("=" * 60)
    print("Black-Hole Multiseed Rerun (Table IV)")
    print("=" * 60)

    # ---- Setup: identical to e2e_attacks.py [1/5]-[2/5] ----
    print("\n[1/4] Loading corpus + embedding...")
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
    e1_band, e0_band, e0_ppb = R.compute_bands(d)

    print("[2/4] Watermarking...")
    wm = np.array([embed_full(E[i], i % 8, U, mu, KEY,
                              e1_band, e0_band, e0_ppb)
                   for i in range(len(E))])

    # ---- Attack generation: identical to e2e_attacks.py Test 1 ----
    print("[3/4] Generating Black-Hole attack vectors...")
    BH_PATH = os.path.expanduser(
        "~/regmark/Black_Hole_Attack_for_Vector_Database/src/attack/centroid.py")
    spec = importlib.util.spec_from_file_location("centroid", BH_PATH)
    bh_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bh_mod)

    km = KMeans(n_clusters=100, random_state=42, n_init=10).fit(wm)
    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    bh_vecs = bh_mod.perturb_centroids(centroids, num_copies=10,
                                       epsilon=0.001, seed=42)
    rng = np.random.default_rng(2026)
    rand_vecs = rng.normal(0, 1, (50, d)).astype(np.float32)
    rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)
    injected = np.vstack([bh_vecs, rand_vecs]).astype(wm.dtype)

    bh_rej = sum(verify_e0(injected[i], U, mu, KEY, e0_band, e0_ppb)[2]
                 == "REJECTED" for i in range(len(injected)))
    print(f"  Injected rejected: {bh_rej}/{len(injected)}")
    res["injected_rejected"] = f"{bh_rej}/{len(injected)}"

    attacked = np.vstack([wm, injected])

    # ---- Per-seed E2E recall: identical loop, now recorded ----
    print("[4/4] Per-seed retrieval (A/B/C)...")

    def load_qa(seed, n=100):
        text2idx = {t: i for i, t in enumerate(texts)}
        qa = [(r["question"], text2idx[r["context"]])
              for r in ds if r["context"] in text2idx]
        r2 = np.random.default_rng(seed)
        r2.shuffle(qa)
        return qa[:n]

    dbn = attacked / (np.linalg.norm(attacked, axis=1,
                                     keepdims=True) + 1e-12)
    wmn = wm / (np.linalg.norm(wm, axis=1, keepdims=True) + 1e-12)

    A, B, C, P = [], [], [], []
    res["per_seed"] = {}
    for seed in SEEDS:
        qa = load_qa(seed)
        qs = [q for q, _ in qa]
        gt_idx = [i for _, i in qa]
        Q = np.asarray(model.encode(qs, normalize_embeddings=True))

        clean_hit = atk_hit = rm_hit = n_poison = 0
        for qi in range(len(qs)):
            qn = Q[qi] / (np.linalg.norm(Q[qi]) + 1e-12)
            ranked = np.argsort(-(dbn @ qn))

            # (A) Clean top-5
            clean_top = np.argsort(-(wmn @ qn))[:5]
            clean_hit += (gt_idx[qi] in clean_top)

            # (B) Attacked top-5
            top5 = ranked[:5]
            atk_hit += (gt_idx[qi] in top5)
            n_poison += sum(1 for j in top5 if j >= len(wm))

            # (C) RagMark-filtered top-5
            verified = []
            for j in ranked:
                if len(verified) >= 5:
                    break
                j = int(j)
                _, _, verd = verify_e0(attacked[j], U, mu, KEY,
                                       e0_band, e0_ppb)
                if verd == "PASS":
                    verified.append(j)
            rm_hit += (gt_idx[qi] in verified)

        A.append(clean_hit); B.append(atk_hit); C.append(rm_hit)
        P.append(n_poison / len(qs))
        res["per_seed"][seed] = {"clean": clean_hit, "attacked": atk_hit,
                                 "ragmark": rm_hit,
                                 "poison_at5": n_poison / len(qs)}
        print(f"  seed {seed}: (A) Clean={clean_hit}%  "
              f"(B) Attack={atk_hit}% poison={n_poison/len(qs):.1f}  "
              f"(C) RagMark={rm_hit}%")

    # ---- Aggregates + the native t-test ----
    A, B, C = np.array(A, float), np.array(B, float), np.array(C, float)
    t_AC, p_AC = stats.ttest_rel(A, C)
    print("\n" + "=" * 60)
    print(f"  (A) Clean   : {A.mean():.1f} +/- {A.std(ddof=1):.1f}  "
          f"(min {A.min():.0f}, max {A.max():.0f})")
    print(f"  (B) Attacked: {B.mean():.1f} +/- {B.std(ddof=1):.1f}  "
          f"poison@5 = {np.mean(P):.1f}")
    print(f"  (C) RagMark : {C.mean():.1f} +/- {C.std(ddof=1):.1f}  "
          f"(min {C.min():.0f}, max {C.max():.0f})")
    print(f"  Paired t-test A vs C: t = {t_AC:.3f}, p = {p_AC:.3f}")
    print(f"  Paper's Table IV claims: 90.2+/-1.6 / 50.0+/-4.1 / "
          f"90.0+/-1.6, poison 4.1, p = 0.37")
    print("=" * 60)

    res["mean_A"] = [float(A.mean()), float(A.std(ddof=1))]
    res["mean_B"] = [float(B.mean()), float(B.std(ddof=1))]
    res["mean_C"] = [float(C.mean()), float(C.std(ddof=1))]
    res["poison_at5_mean"] = float(np.mean(P))
    res["ttest_A_vs_C"] = {"t": float(t_AC), "p": float(p_AC)}

    with open("results/blackhole_multiseed.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("\nSaved results/blackhole_multiseed.json")


if __name__ == "__main__":
    main()
