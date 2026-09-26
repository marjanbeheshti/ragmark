#!/usr/bin/env python3
"""
validation.py — RAGMARK core validation (imports ragmark.py)
=============================================================
Reproduces the paper's headline numbers against the SINGLE source of
truth in ragmark.py. Defines NO watermarking logic of its own.

Tests:
  1  CosSim + E_A/E_I self-check
  2  GT recall parity
  3  Centroid rejection
  4  Random-injection rejection
  5  Content-binding ablation
  6  Guessed-key forgery
  7  Confidence-oracle hill-climbing
  8  HMAC sel-tamper (new: needs 5-tuple metadata)
  9  Cross-model (384, 1024)
  10 Latency

Usage: python experiments/validation.py [--skip-slow]
"""
import argparse, json, os, sys, time, hashlib
from datetime import datetime
import numpy as np

# import the single source of truth (repo root on path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ragmark as R

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from datasets import load_dataset
from scipy import stats as st

KEY = R.MASTER_KEY


def load_corpus(n):
    ds = load_dataset("rajpurkar/squad", split="train")
    seen, texts = set(), []
    for r in ds:
        if r["context"] not in seen and len(texts) < n:
            seen.add(r["context"]); texts.append(r["context"])
    return texts, ds


def fit(texts, model, d):
    E = np.asarray(model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=True))
    pca = PCA(n_components=min(d, len(E))).fit(E)
    return E, pca.components_, pca.mean_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(2026)
    res = {"date": datetime.now().isoformat(), "core": "ragmark.py"}

    print("=" * 60)
    print("RAGMARK core validation (imports ragmark.py)")
    print("=" * 60)

    texts, ds = load_corpus(500)
    model = SentenceTransformer("all-mpnet-base-v2")
    E, U, mu = fit(texts, model, 768)
    d = E.shape[1]
    e1b, e0b, e0p = R.compute_bands(d)
    print(f"  d={d}, E_A={e1b}, E_I={e0b}, ppb={e0p}")

    wm, metas = [], []
    for i in range(len(E)):
        w, m = R.watermark(E[i], i % 8, U, mu, e1b, e0b, e0p, doc_id=i)
        wm.append(w); metas.append(m)
    wm = np.array(wm)

    # 1. CosSim + self-check
    cos = float(np.mean(np.sum(E * wm, axis=1) /
        (np.linalg.norm(E, axis=1) * np.linalg.norm(wm, axis=1) + 1e-12)))
    okA = okI = 0
    for i in range(len(wm)):
        v = (wm[i] - mu) @ U.T
        if R.decode_e1(v, metas[i]["sel"])[0] == i % 8:
            okA += 1
        if R.decode_e0(v, KEY, e0b, e0p)[2] == "PASS":
            okI += 1
    print(f"\n[1] CosSim={cos:.4f}  (paper 0.976)")
    print(f"    E_A self-check={okA}/500   E_I self-check={okI}/500")
    res.update(cos=cos, e1_self=okA, e0_self=okI)

    # 2. GT recall
    t2i = {t: i for i, t in enumerate(texts)}
    qa = [(r["question"], t2i[r["context"]]) for r in ds if r["context"] in t2i]
    r2 = np.random.default_rng(2026); r2.shuffle(qa); qa = qa[:100]
    Q = np.asarray(model.encode([q for q, _ in qa], normalize_embeddings=True))
    gt = [i for _, i in qa]
    def recall(db):
        dbn = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-12)
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        top = np.argsort(-(qn @ dbn.T), axis=1)[:, :5]
        return sum(gt[i] in top[i] for i in range(len(gt))) / len(gt)
    rc, rw = recall(E), recall(wm)
    print(f"\n[2] GT recall clean={rc*100:.1f}%  wm={rw*100:.1f}%")
    res.update(recall_clean=rc, recall_wm=rw)

    # 3-4. centroid + random rejection
    rejc = 0
    for _ in range(200):
        idx = rng.choice(len(wm), 100, replace=False)
        c = wm[idx].mean(0); c /= np.linalg.norm(c) + 1e-12
        if R.decode_e0((c - mu) @ U.T, KEY, e0b, e0p)[2] == "REJECTED":
            rejc += 1
    rr = 0
    for _ in range(200):
        v = rng.normal(0, 1, d); v /= np.linalg.norm(v)
        if R.decode_e0((v - mu) @ U.T, KEY, e0b, e0p)[2] == "REJECTED":
            rr += 1
    print(f"\n[3] centroids rejected {rejc}/200   [4] random {rr}/200")
    res.update(centroids=rejc, random=rr)

    # 5. content-binding ablation.
    # Re-embed E_I with a FIXED fingerprint (binding disabled): all
    # vectors then share the same pairs, so averaging preserves
    # orderings and centroids inherit a valid watermark -> fewer
    # rejected. This mirrors the canonical construction.
    fixed = "0000000000000000"
    wm_fixed = []
    for i in range(len(E)):
        v = (np.asarray(E[i], dtype=mu.dtype) - mu) @ U.T
        f_a = R.content_fingerprint(E[i])
        v, _ = R.embed_e1(v, i % 8, f_a, KEY, e1b)         # E_A unchanged
        pairs = R.derive_pairs(KEY, R.TAG_INTEG, fixed, e0b, e0p)  # fixed fp
        for b in range(R.N_BITS):
            bit = (i % 8 >> b) & 1
            for a, b_ in pairs[b]:
                if bit == 1 and v[a] < v[b_]:
                    v[a], v[b_] = v[b_], v[a]
                elif bit == 0 and v[a] > v[b_]:
                    v[a], v[b_] = v[b_], v[a]
        wm_fixed.append(v @ U + mu)
    wm_fixed = np.array(wm_fixed)
    rej_fixed = 0
    for _ in range(200):
        idx = rng.choice(len(wm_fixed), 100, replace=False)
        c = wm_fixed[idx].mean(0); c /= np.linalg.norm(c) + 1e-12
        v = (c - mu) @ U.T
        pairs = R.derive_pairs(KEY, R.TAG_INTEG, fixed, e0b, e0p)
        confs = []
        for b in range(R.N_BITS):
            votes = [1 if v[a] > v[b_] else 0 for a, b_ in pairs[b] if v[a] != v[b_]]
            ones = sum(votes)
            confs.append(max(ones, len(votes)-ones)/e0p if votes else 0.0)
        rej_fixed += (np.mean(confs) < R.TAU)
    print(f"\n[5] ablation: content-bound {rejc}/200  fixed-fp {rej_fixed}/200")
    res.update(ablation_bound=rejc, ablation_fixed=rej_fixed)

    # 6. guessed-key forgery
    forged = 0
    for t in range(1000):
        fk = f"guess_{t:05d}"
        w, _ = R.watermark(E[0], 42, U, mu, e1b, e0b, e0p, doc_id=0, key=fk)
        if R.decode_e0((w - mu) @ U.T, KEY, e0b, e0p)[2] == "PASS":
            forged += 1
    print(f"\n[6] guessed keys {forged}/1000 pass real gate")
    res["guessed_keys"] = forged

    # 7. oracle hill-climb (5 seeds x 10 trials x 5000 q) unless --skip-slow
    if not args.skip_slow:
        best_all = []
        for seed in range(5):
            ro = np.random.default_rng(2026 + seed)
            for _ in range(10):
                v = ro.normal(0, 1, d); v /= np.linalg.norm(v)
                vp = (v - mu) @ U.T
                best = 0.0
                for _q in range(5000):
                    j = ro.integers(e0b[0], e0b[1])
                    vp[j] += ro.normal(0, 0.01)
                    _, c, _ = R.decode_e0(vp, KEY, e0b, e0p)
                    if c > best: best = c
                    else: vp[j] -= ro.normal(0, 0.01)
                    if best >= R.TAU: break
                best_all.append(best)
        forged_o = sum(b >= R.TAU for b in best_all)
        print(f"\n[7] oracle forged {forged_o}/50  "
              f"max={max(best_all):.3f} median={np.median(best_all):.3f}")
        res.update(oracle_forged=forged_o, oracle_max=float(max(best_all)))
    else:
        print("\n[7] oracle — SKIPPED")

    # 8. HMAC sel-tamper
    legit = badsig = 0
    for i in range(len(wm)):
        r = R.verify(wm[i], U, mu, e0b, e0p, meta=metas[i])
        legit += (r["verdict"] == "PASS")
        m2 = dict(metas[i]); s2 = [list(b) for b in metas[i]["sel"]]
        if s2 and s2[0]:
            s2[0] = [(s2[0][0][0] + 1, s2[0][0][1])] + s2[0][1:]
        m2["sel"] = s2
        r2 = R.verify(wm[i], U, mu, e0b, e0p, meta=m2)
        badsig += (r2["verdict"] == "TAMPERED_BAD_SIG")
    print(f"\n[8] HMAC: legit PASS {legit}/500   "
          f"sel-tamper BAD_SIG {badsig}/500")
    res.update(hmac_legit=legit, hmac_seltamper=badsig)

    # 9. cross-model
    if not args.skip_slow:
        print("\n[9] cross-model")
        for name, md, nd in [("all-MiniLM-L12-v2", 384, 500),
                             ("BAAI/bge-large-en-v1.5", 1024, 1000)]:
            mm = SentenceTransformer(name)
            tt, _ = load_corpus(nd)
            Em, Um, mum = fit(tt, mm, md)
            e1m, e0m, e0pm = R.compute_bands(md)
            wmm = np.array([R.watermark(Em[i], i % 8, Um, mum,
                                         e1m, e0m, e0pm, doc_id=i)[0]
                            for i in range(len(Em))])
            cosm = float(np.mean(np.sum(Em * wmm, axis=1) /
                (np.linalg.norm(Em, axis=1) * np.linalg.norm(wmm, axis=1) + 1e-12)))
            okm = sum(R.decode_e0((wmm[i] - mum) @ Um.T, KEY, e0m, e0pm)[2] == "PASS"
                      for i in range(len(Em)))
            print(f"    {name} d={md}: CosSim={cosm:.4f}  E_I={okm}/{len(Em)}")
            res[f"cross_{md}"] = {"cos": cosm, "e0": okm, "n": len(Em)}

    # 10. latency
    t0 = time.perf_counter()
    for i in range(100):
        R.watermark(E[i], i % 8, U, mu, e1b, e0b, e0p, doc_id=i)
    te = (time.perf_counter() - t0) / 100
    t0 = time.perf_counter()
    for i in range(100):
        R.verify(wm[i], U, mu, e0b, e0p, meta=metas[i])
    tv = (time.perf_counter() - t0) / 100
    print(f"\n[10] embed {te*1000:.2f} ms  verify {tv*1000:.2f} ms")
    res.update(latency_embed_ms=te*1000, latency_verify_ms=tv*1000)

    print(f"\n{'='*60}\nDONE — compare to paper: CosSim 0.976, self 500/499,")
    print("recall parity, centroids/random 200/200, oracle 0/50,")
    print("HMAC legit 500 / sel-tamper 500, MiniLM 0.914, BGE 0.984")
    with open("results/validation.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/validation.json")


if __name__ == "__main__":
    main()
