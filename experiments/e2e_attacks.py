#!/usr/bin/env python3
"""
Dual-Band E2E: Black-Hole + PoisonedRAG + LLM
================================================
Tests the final dual-band design (greedy selection + full-vector fp)
against real attacks:
  1. Real Black-Hole attack (5 seeds, recall recovery)
  2. PoisonedRAG passage rejection (500 passages)
  3. PoisonedRAG LLM trick rate (optional, --with-llm)

Usage:
    python experiments/dualband_e2e_attacks.py
    python experiments/dualband_e2e_attacks.py --with-llm
    python experiments/dualband_e2e_attacks.py --with-llm --llm-model mistral:7b
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import string
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
from sklearn.cluster import KMeans
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


def embed_full(v_raw, source_id, U, mu, key, e1_band, e0_band, e0_ppb):
    """Shim -> ragmark.watermark. Returns (w_raw, (fp, sel)) to match
    the original signature used by the attack code."""
    w_raw, meta = R.watermark(v_raw, source_id, U, mu,
                              e1_band, e0_band, e0_ppb,
                              doc_id=source_id, key=key)
    return w_raw


def verify_e0(v_raw, U, mu, key, e0_band, e0_ppb):
    """Shim -> ragmark E_I decode. Returns (sid, conf, verdict)."""
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T
    return R.decode_e0(v_pc, key, e0_band, e0_ppb)


def ollama_generate(prompt, model_name, url):
    import urllib.request
    data = json.dumps({
        "model": model_name, "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 500}
    }).encode()
    req = urllib.request.Request(
        f"{url}/api/generate", data=data,
        headers={"Content-Type": "application/json"})
    try:
        raw = json.loads(
            urllib.request.urlopen(req, timeout=60).read())["response"]
        return re.sub(r'<think>.*?</think>', '', raw,
                      flags=re.DOTALL).strip()
    except Exception as e:
        return ""


def normalize_answer(s):
    s = re.sub(r'\b(a|an|the)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--llm-model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    results = {"date": datetime.now().isoformat(),
               "design": "dual-band + greedy + full-vec-fp"}

    print("=" * 60)
    print("Dual-Band E2E: Black-Hole + PoisonedRAG + LLM")
    print("=" * 60)

    # ---- Setup ----
    print("\n[1/5] Loading corpus + embedding...")
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

    # Watermark
    print("[2/5] Watermarking...")
    wm = np.array([embed_full(E[i], i % 8, U, mu, KEY,
                               e1_band, e0_band, e0_ppb)
                    for i in range(len(E))])
    cos = float(np.mean(np.sum(E * wm, axis=1) /
        (np.linalg.norm(E, axis=1) * np.linalg.norm(wm, axis=1)
         + 1e-12)))
    ok = sum(verify_e0(wm[i], U, mu, KEY, e0_band, e0_ppb)[2] == "PASS"
             for i in range(len(wm)))
    print(f"  CosSim={cos:.4f}  E0' self-check={ok}/{len(wm)}")

    # ================================================================
    # TEST 1: Real Black-Hole Attack (5 seeds)
    # ================================================================
    print("\n[3/5] Black-Hole E2E (5 seeds)...")

    # Load Black-Hole code
    BH_PATH = os.path.expanduser(
        "~/regmark/Black_Hole_Attack_for_Vector_Database/src/attack/centroid.py")
    spec = importlib.util.spec_from_file_location("centroid", BH_PATH)
    bh_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bh_mod)

    # Generate attack vectors
    km = KMeans(n_clusters=100, random_state=42, n_init=10).fit(wm)
    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    bh_vecs = bh_mod.perturb_centroids(centroids, num_copies=10,
                                        epsilon=0.001, seed=42)
    rng = np.random.default_rng(2026)
    rand_vecs = rng.normal(0, 1, (50, d)).astype(np.float32)
    rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)
    injected = np.vstack([bh_vecs, rand_vecs]).astype(wm.dtype)

    # Rejection check
    bh_rej = sum(verify_e0(injected[i], U, mu, KEY, e0_band, e0_ppb)[2]
                 == "REJECTED" for i in range(len(injected)))
    print(f"  Injected rejected: {bh_rej}/{len(injected)}")

    # Build attacked database
    attacked = np.vstack([wm, injected])
    atk_texts = list(texts) + ["ADVERSARIAL"] * len(injected)

    # Per-seed E2E recall
    def load_qa(seed, n=100):
        text2idx = {t: i for i, t in enumerate(texts)}
        qa = [(r["question"], text2idx[r["context"]])
              for r in ds if r["context"] in text2idx]
        r2 = np.random.default_rng(seed)
        r2.shuffle(qa)
        return qa[:n]

    for seed in [2026, 2027, 2028, 2029, 2030]:
        qa = load_qa(seed)
        qs = [q for q, _ in qa]
        gt_idx = [i for _, i in qa]
        Q = np.asarray(model.encode(qs, normalize_embeddings=True))
        dbn = attacked / (np.linalg.norm(attacked, axis=1,
                                          keepdims=True) + 1e-12)

        clean_hit = atk_hit = rm_hit = n_poison = 0
        for qi in range(len(qs)):
            qn = Q[qi] / (np.linalg.norm(Q[qi]) + 1e-12)
            ranked = np.argsort(-(dbn @ qn))

            # (A) Clean top-5
            clean_sims = (wm / (np.linalg.norm(wm, axis=1,
                                 keepdims=True) + 1e-12)) @ qn
            clean_top = np.argsort(-clean_sims)[:5]
            clean_hit += (gt_idx[qi] in clean_top)

            # (B) Attacked top-5
            top5 = ranked[:5]
            atk_hit += (gt_idx[qi] in top5)
            n_poison += sum(1 for j in top5 if j >= len(wm))

            # (C) RAGMARK-filtered top-5
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

        print(f"  seed {seed}: (A) Clean={clean_hit}%  "
              f"(B) Attack={atk_hit}% poison={n_poison/len(qs):.1f}  "
              f"(C) RAGMARK={rm_hit}%")

    results["blackhole_rejected"] = bh_rej

    # ================================================================
    # TEST 2: PoisonedRAG Rejection
    # ================================================================
    print("\n[4/5] PoisonedRAG rejection...")
    prag_path = os.path.expanduser(
        "~/regmark/PoisonedRAG/results/adv_targeted_results/nq.json")
    prag = json.load(open(prag_path))
    poison_texts = []
    for k in prag:
        poison_texts.extend(prag[k]["adv_texts"])
    print(f"  Loaded {len(poison_texts)} adversarial passages")
    Pp = np.asarray(model.encode(poison_texts, normalize_embeddings=True))
    rej_p = sum(verify_e0(Pp[i], U, mu, KEY, e0_band, e0_ppb)[2]
                == "REJECTED" for i in range(len(Pp)))
    confs_p = [verify_e0(Pp[i], U, mu, KEY, e0_band, e0_ppb)[1]
               for i in range(len(Pp))]
    print(f"  Rejected: {rej_p}/{len(Pp)}")
    print(f"  Confidence: mean={np.mean(confs_p):.3f} "
          f"max={np.max(confs_p):.3f}")
    results["poisonedrag_rejected"] = rej_p
    results["poisonedrag_conf_max"] = float(np.max(confs_p))

    # ================================================================
    # TEST 3: PoisonedRAG LLM Trick Rate (optional)
    # ================================================================
    if args.with_llm:
        print(f"\n[5/5] PoisonedRAG LLM ({args.llm_model})...")

        # Build attacked DB with PoisonedRAG vectors
        Pp = Pp.astype(wm.dtype)
        atk_db = np.vstack([wm, Pp])
        atk_db_texts = list(texts) + poison_texts
        test_ids = list(prag.keys())

        res_a = {"correct": 0, "tricked": 0, "total": 0}
        res_b = {"correct": 0, "tricked": 0, "total": 0}
        res_c = {"correct": 0, "tricked": 0, "total": 0}

        for qi, tid in enumerate(test_ids):
            question = prag[tid]["question"]
            correct_ans = prag[tid]["correct answer"]
            wrong_ans = prag[tid]["incorrect answer"]
            q_vec = np.asarray(
                model.encode([question], normalize_embeddings=True))[0]

            dbn_p = atk_db / (np.linalg.norm(atk_db, axis=1,
                                              keepdims=True) + 1e-12)
            qn = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            ranked = np.argsort(-(dbn_p @ qn))

            # (A) Clean
            clean_sims = (wm / (np.linalg.norm(wm, axis=1,
                                 keepdims=True) + 1e-12)) @ qn
            top_c = np.argsort(-clean_sims)[:5]
            ctx_a = [texts[j] for j in top_c]

            # (B) Attacked
            top_b = ranked[:5]
            ctx_b = [atk_db_texts[j] for j in top_b]

            # (C) RAGMARK filtered
            verified = []
            for j in ranked:
                if len(verified) >= 5:
                    break
                j = int(j)
                _, _, verd = verify_e0(atk_db[j], U, mu, KEY,
                                       e0_band, e0_ppb)
                if verd == "PASS":
                    verified.append(j)
            ctx_c = [atk_db_texts[j] for j in verified]

            for label, ctx, res in [("A", ctx_a, res_a),
                                    ("B", ctx_b, res_b),
                                    ("C", ctx_c, res_c)]:
                prompt = (f"Answer based ONLY on these documents. "
                         f"Short answer.\n\n" +
                         "\n\n".join(f"Doc {i+1}: {c}"
                                    for i, c in enumerate(ctx)) +
                         f"\n\nQ: {question}\nA:")
                ans = ollama_generate(prompt, args.llm_model,
                                      args.ollama_url)
                res["total"] += 1
                if normalize_answer(correct_ans) in normalize_answer(ans):
                    res["correct"] += 1
                if normalize_answer(wrong_ans) in normalize_answer(ans):
                    res["tricked"] += 1

            if (qi + 1) % 10 == 0:
                print(f"  [{qi+1}/{len(test_ids)}] "
                      f"B_tricked={res_b['tricked']}/{res_b['total']} "
                      f"C_tricked={res_c['tricked']}/{res_c['total']}")

        print(f"\n  {'Condition':<30s} {'Correct%':>10s} {'Tricked%':>10s}")
        print("  " + "-" * 52)
        for name, res in [("(A) Clean baseline", res_a),
                          ("(B) Attacked, no verif.", res_b),
                          ("(C) Attacked + RAGMARK", res_c)]:
            c = 100 * res["correct"] / max(res["total"], 1)
            t = 100 * res["tricked"] / max(res["total"], 1)
            print(f"  {name:<30s} {c:>9.1f}% {t:>9.1f}%")

        results["llm_model"] = args.llm_model
        results["llm_a"] = res_a
        results["llm_b"] = res_b
        results["llm_c"] = res_c
    else:
        print("\n[5/5] PoisonedRAG LLM — SKIPPED (use --with-llm)")

    print(f"\n{'='*60}")
    with open("results/e2e_attacks.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved results/e2e_attacks.json")


if __name__ == "__main__":
    main()
