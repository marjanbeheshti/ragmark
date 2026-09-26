#!/usr/bin/env python3
"""
Experiment 1: Domain-Separation Revalidation
=============================================
Patches the two seed derivations to the domain-separated construction:
    E_A seed:  H(K || "attr"  || f_A)
    E_I seed:  H(K || "integ" || f_I)
    (HMAC key: K' = H(K || "hmac"), if used)

Then re-runs the full validation and prints old-vs-new side by side.
The tag only relabels independent streams, so every metric should be
statistically identical (small ties/rounding differences possible).

This script is SELF-CONTAINED: it redefines the greedy + full-vec
functions with the domain tag inline, so it does not depend on whether
poisonedrag_test.py has been patched. It only imports P for the corpus
fingerprint and pair-derivation primitives, which it wraps.

Usage:
    python experiments/exp1_domain_separation.py
    python experiments/exp1_domain_separation.py --skip-slow
"""
import argparse
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

# Domain-separation tags
TAG_ATTR = "attr"
TAG_INTEG = "integ"


def compute_bands(d):
    e1_lo = max(1, int(round(0.04 * d)))
    e1_hi = int(round(0.2 * d))
    e0_lo = e1_hi
    e0_hi = int(round(0.5 * d))
    e0_ppb = min(10, (e0_hi - e0_lo) // (2 * N_BITS))
    return (e1_lo, e1_hi), (e0_lo, e0_hi), e0_ppb


# =====================================================================
# Pair derivation WITH domain separation
# =====================================================================
def derive_pairs_tagged(key, tag, fingerprint, band, ppb, n_bits=N_BITS):
    """Domain-separated pair derivation: seed = H(key || tag || fp)."""
    seed_val = int(hashlib.sha256(
        (key + "|" + tag + "|" + fingerprint).encode()).hexdigest(),
        16) % (2**32)
    rng = np.random.default_rng(seed_val)
    indices = list(range(band[0], band[1]))
    rng.shuffle(indices)
    pairs = []
    pos = 0
    for b in range(n_bits):
        bit_pairs = []
        for _ in range(ppb):
            bit_pairs.append((indices[pos], indices[pos + 1]))
            pos += 2
        pairs.append(bit_pairs)
    return pairs


# =====================================================================
# E_A: greedy, tagged seed
# =====================================================================
def greedy_select_e1(v_pc, source_id, fingerprint, key, band,
                     tag=TAG_ATTR, ppb=PPB_E1, n_bits=N_BITS, delta=DELTA):
    w = v_pc.copy()
    seed_val = int(hashlib.sha256(
        (key + "|" + tag + "|" + fingerprint).encode()).hexdigest(),
        16) % (2**32)
    rng = np.random.default_rng(seed_val)
    all_indices = list(range(band[0], band[1]))
    rng.shuffle(all_indices)
    committed = set()
    sel = []
    for b in range(n_bits):
        bit_val = (source_id >> b) & 1
        avail = [i for i in all_indices if i not in committed]
        ncand = len(avail) // 2
        cands = []
        for ci in range(ncand):
            a, b_ = avail[2 * ci], avail[2 * ci + 1]
            gap = (w[a] - w[b_]) if bit_val == 1 else (w[b_] - w[a])
            cands.append((gap, a, b_))
        cands.sort(key=lambda x: -x[0])
        chosen = []
        for gap, a, b_ in cands[:ppb]:
            committed.add(a); committed.add(b_)
            chosen.append((a, b_))
            if gap < 0:
                mid = (w[a] + w[b_]) / 2
                if bit_val == 1:
                    w[a], w[b_] = mid + delta, mid - delta
                else:
                    w[a], w[b_] = mid - delta, mid + delta
        sel.append(chosen)
    return w, sel


def decode_e1(v_pc, sel, ppb=PPB_E1, n_bits=N_BITS):
    bits, confs = [], []
    for b in range(n_bits):
        votes = []
        for a, b_ in sel[b]:
            if v_pc[a] == v_pc[b_]:
                continue
            votes.append(1 if v_pc[a] > v_pc[b_] else 0)
        if not votes:
            bits.append(0); confs.append(0.0)
        else:
            ones = sum(votes)
            bits.append(1 if ones > len(votes) / 2 else 0)
            confs.append(max(ones, len(votes) - ones) / ppb)
    sid = sum(bit << j for j, bit in enumerate(bits))
    return sid, float(np.mean(confs))


# =====================================================================
# E_I: swap, tagged seed, full-vector fingerprint
# =====================================================================
def full_fp(v_pc):
    c = np.round(v_pc, 2) + 0.0
    return hashlib.sha256(str(tuple(np.sort(c))).encode()).hexdigest()[:16]


def e0_embed(v_pc, source_id, key, band, ppb, tag=TAG_INTEG, n_bits=N_BITS):
    w = v_pc.copy()
    fp = full_fp(v_pc)
    pairs = derive_pairs_tagged(key, tag, fp, band, ppb, n_bits)
    for b in range(n_bits):
        bit_val = (source_id >> b) & 1
        for a, b_ in pairs[b]:
            if bit_val == 1 and w[a] < w[b_]:
                w[a], w[b_] = w[b_], w[a]
            elif bit_val == 0 and w[a] > w[b_]:
                w[a], w[b_] = w[b_], w[a]
    return w


def e0_decode(v_pc, key, band, ppb, tag=TAG_INTEG, n_bits=N_BITS, tau=TAU):
    fp = full_fp(v_pc)
    pairs = derive_pairs_tagged(key, tag, fp, band, ppb, n_bits)
    bits, confs = [], []
    for b in range(n_bits):
        votes = []
        for a, b_ in pairs[b]:
            if v_pc[a] == v_pc[b_]:
                continue
            votes.append(1 if v_pc[a] > v_pc[b_] else 0)
        if not votes:
            bits.append(0); confs.append(0.0)
        else:
            ones = sum(votes)
            bits.append(1 if ones > len(votes) / 2 else 0)
            confs.append(max(ones, len(votes) - ones) / ppb)
    sid = sum(bit << j for j, bit in enumerate(bits))
    conf = float(np.mean(confs))
    return sid, conf, "PASS" if conf >= tau else "REJECTED"


def embed_full(v_raw, sid, U, mu, key, e1b, e0b, e0ppb):
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T
    e1_fp = R.content_fingerprint(v_raw)
    v_pc, sel = greedy_select_e1(v_pc, sid, e1_fp, key, e1b)
    v_pc = e0_embed(v_pc, sid, key, e0b, e0ppb)
    return v_pc @ U + mu, (e1_fp, sel)


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-slow", action="store_true")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(2026)
    res = {"date": datetime.now().isoformat(),
           "design": "domain-separated (attr/integ tags)"}

    print("=" * 64)
    print("Experiment 1: Domain-Separation Revalidation")
    print("Seeds: E_A=H(K||'attr'||f_A)  E_I=H(K||'integ'||f_I)")
    print("=" * 64)

    print("\n[SETUP] corpus (500 docs)...")
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
    e1b, e0b, e0ppb = compute_bands(d)
    print(f"  d={d}, E_A={e1b}, E_I={e0b}")

    wm, metas = [], []
    for i in range(len(E)):
        w, m = embed_full(E[i], i % 8, U, mu, KEY, e1b, e0b, e0ppb)
        wm.append(w); metas.append(m)
    wm = np.array(wm)

    # ---- Core metrics (compare against un-tagged baseline) ----
    print("\n[1] Self-check + CosSim")
    cos = float(np.mean(np.sum(E * wm, axis=1) /
        (np.linalg.norm(E, axis=1) * np.linalg.norm(wm, axis=1) + 1e-12)))
    ok_e1 = sum(decode_e1((wm[i] - mu) @ U.T, metas[i][1])[0] == i % 8
                for i in range(len(wm)))
    ok_e0 = sum(e0_decode((wm[i] - mu) @ U.T, KEY, e0b, e0ppb)[2] == "PASS"
                for i in range(len(wm)))
    print(f"  CosSim={cos:.4f}  (baseline 0.976)")
    print(f"  E_A self-check={ok_e1}/500  (baseline 500)")
    print(f"  E_I self-check={ok_e0}/500  (baseline 499)")
    res["cos"] = cos; res["e1_self"] = ok_e1; res["e0_self"] = ok_e0

    # ---- Independence check: are the two streams actually different? ----
    print("\n[2] Stream-independence check (tagged vs untagged seeds)")
    # For 100 vectors, confirm E_A pairs != E_I pairs and tagged != untagged
    n_overlap = 0
    for i in range(100):
        v_pc = (wm[i] - mu) @ U.T
        fp_i = full_fp(v_pc)
        # tagged E_I pairs
        pi = derive_pairs_tagged(KEY, TAG_INTEG, fp_i, e0b, e0ppb)
        # untagged (old) E_I pairs via P module
        po = R.derive_pairs(KEY, R.TAG_INTEG, fp_i, e0b, e0ppb, N_BITS)
        flat_i = set((a, b) for bit in pi for a, b in bit)
        flat_o = set((a, b) for bit in po for a, b in bit)
        if flat_i == flat_o:
            n_overlap += 1
    print(f"  tagged E_I pairs identical to untagged: {n_overlap}/100")
    print(f"  (expect 0 — the tag must change the derived pairs)")
    res["stream_overlap"] = n_overlap

    # ---- Centroid rejection ----
    print("\n[3] Centroid rejection")
    rej_c = 0
    for _ in range(200):
        idx = rng.choice(len(wm), 100, replace=False)
        c = wm[idx].mean(0); c /= np.linalg.norm(c) + 1e-12
        if e0_decode((c - mu) @ U.T, KEY, e0b, e0ppb)[2] == "REJECTED":
            rej_c += 1
    print(f"  {rej_c}/200 rejected  (baseline 200)")
    res["centroids"] = rej_c

    # ---- KS invisibility (E_A band + E_I band) ----
    print("\n[4] KS invisibility")
    from scipy import stats as st
    clean_pc = np.array([(E[i] - mu) @ U.T for i in range(len(E))])
    wm_pc = np.array([(wm[i] - mu) @ U.T for i in range(len(wm))])
    for label, lo, hi in [("E_A", e1b[0], e1b[1]), ("E_I", e0b[0], e0b[1])]:
        _, p = st.ks_2samp(clean_pc[:, lo:hi].ravel(),
                           wm_pc[:, lo:hi].ravel())
        rej = sum(st.ks_2samp(clean_pc[:, pc], wm_pc[:, pc])[1] < 0.05
                  for pc in range(lo, hi))
        print(f"  {label} band: KS p={p:.4f}  per-PC reject={rej}/{hi-lo}")
        res[f"ks_{label}_p"] = float(p)

    # ---- Guessed-key forgery ----
    print("\n[5] Guessed-key forgery")
    forged = 0
    for t in range(1000):
        fk = f"guess_{t:05d}"
        v_pc = (E[0] - mu) @ U.T
        fp0 = R.content_fingerprint(E[0])
        v_pc, _ = greedy_select_e1(v_pc, 42, fp0, fk, e1b)
        v_pc = e0_embed(v_pc, 42, fk, e0b, e0ppb)
        w = v_pc @ U + mu
        if e0_decode((w - mu) @ U.T, KEY, e0b, e0ppb)[2] == "PASS":
            forged += 1
    print(f"  {forged}/1000 pass real gate  (baseline 0)")
    res["guessed_keys"] = forged

    # ---- Scale (optional) ----
    if not args.skip_slow:
        print("\n[6] Scale (10k)")
        seen10, t10 = set(), []
        for r in ds:
            if r["context"] not in seen10 and len(t10) < 10000:
                seen10.add(r["context"]); t10.append(r["context"])
        E10 = np.asarray(model.encode(t10, normalize_embeddings=True,
                                       batch_size=128,
                                       show_progress_bar=True))
        pca10 = PCA(n_components=min(d, len(E10))).fit(E10)
        U10, mu10 = pca10.components_, pca10.mean_
        wm10 = np.array([embed_full(E10[i], i % 8, U10, mu10, KEY,
                                     e1b, e0b, e0ppb)[0]
                          for i in range(len(E10))])
        cos10 = float(np.mean(np.sum(E10 * wm10, axis=1) /
            (np.linalg.norm(E10, axis=1) * np.linalg.norm(wm10, axis=1)
             + 1e-12)))
        ok10 = sum(e0_decode((wm10[i] - mu10) @ U10.T, KEY, e0b, e0ppb)[2]
                   == "PASS" for i in range(len(wm10)))
        print(f"  CosSim={cos10:.4f}  self-check={ok10}/10000  "
              f"(baseline 0.947, 9991)")
        res["scale_cos"] = cos10; res["scale_self"] = ok10
    else:
        print("\n[6] Scale — SKIPPED")

    print(f"\n{'='*64}")
    print("VERDICT: metrics should match the un-tagged baseline within")
    print("rounding; stream_overlap MUST be 0 (tag changed the pairs).")
    print(f"{'='*64}")
    with open("results/domain_separation.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/domain_separation.json")


if __name__ == "__main__":
    main()
