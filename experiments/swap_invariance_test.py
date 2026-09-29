#!/usr/bin/env python3
"""
swap_invariance_test.py — Remark 1 (Swap-preserving edits)
==========================================================
Tests the Remark 1 claim: a value-permuting edit that keeps the sorted
multiset unchanged leaves f_I invariant and does NOT reshuffle the keyed
pairing (which is derived from H(K || "integ" || f_I)). Absent the key K,
such an edit hits the pairs at random and degrades to in-band noise, which
E_I DETECTS (rejects) rather than silently admits.

Two permutation variants are measured, both preserving the multiset:

  (A) FULL-VECTOR permutation: permute all d PC coords with a random
      permutation. The multiset over ALL coords is preserved, so f_I is
      byte-identical and the pairing is unchanged. This is the strongest
      form of the Remark-1 attacker (multiset-invariance fully available).

  (B) E_I-BAND permutation: permute only the coords inside the E_I band
      [l2,h2). The full multiset is still preserved (we only move values
      that were already in the band), so f_I stays invariant here too.

For each, we watermark, apply the permutation, run E_I detect, and report
the reject rate. Expectation per Remark 1: ~100% reject (0 pass), because
the key-secret pairing means a random value-permutation cannot preserve the
encoded orderings.

Control: the IDENTITY "permutation" (no change) must PASS ~100%, confirming
the detector is not simply rejecting everything.

Run:
  python experiments/swap_invariance_test.py
Writes results/swap_invariance.json
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ragmark as R


def build_basis(embeddings):
    """PCA basis (U, mu) from a corpus of raw embeddings.

    U must be a full d x d orthonormal matrix so that v_pc has all d
    coordinates. With full_matrices=True, Vt is d x d: the first
    rank(X) rows are the true PCs and the remainder are an orthonormal
    completion of the null space — still a valid orthonormal basis, so
    the round trip v_pc @ U + mu is exact and the E_I band is well
    defined. (A truncated basis would place band coords in the null
    space and break decode.)
    """
    mu = embeddings.mean(axis=0)
    X = embeddings - mu
    d = X.shape[1]
    _, _, Vt = np.linalg.svd(X, full_matrices=True)  # Vt is d x d
    assert Vt.shape == (d, d), f"basis not full: {Vt.shape}"
    return Vt, mu  # U = Vt ; v_pc = (v-mu) @ U.T


def load_corpus(n_docs, seed=2026, model_name="all-mpnet-base-v2"):
    """Real SQuAD embeddings if sentence-transformers is available,
    else a deterministic synthetic corpus so the test still runs."""
    try:
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
        ds = load_dataset("squad", split="train")
        seen, texts = set(), []
        for r in ds:
            c = r["context"]
            if c not in seen:
                seen.add(c); texts.append(c)
            if len(texts) >= n_docs:
                break
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, convert_to_numpy=True,
                           normalize_embeddings=True)
        return emb.astype(np.float64), "squad/" + model_name
    except Exception as e:
        rng = np.random.default_rng(seed)
        # 768-dim; use >= d docs so the PCA basis is full-rank / well
        # conditioned (mirrors the real 500-doc case, where the paper's
        # basis is likewise full via the raw embedding geometry).
        d = 768
        m = max(n_docs, d + 50)
        base = rng.standard_normal((m, d))
        L = rng.standard_normal((m, 40)) @ rng.standard_normal((40, d))
        emb = base * 0.3 + L
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype(np.float64), f"synthetic(d={d},m={m}) [{type(e).__name__}]"


def run(n_docs=500, seed=2026):
    emb, source = load_corpus(n_docs, seed)
    d = emb.shape[1]
    U, mu = build_basis(emb)
    e1_band, e0_band, e0_ppb = R.compute_bands(d)
    l2, h2 = e0_band
    rng = np.random.default_rng(seed)

    n = min(n_docs, emb.shape[0])
    results = {
        "identity_pass": 0,          # control: unchanged -> PASS
        "fullvec_perm_reject": 0,    # (A) full-vector permutation -> REJECT
        "band_perm_reject": 0,       # (B) E_I-band permutation -> REJECT
        "fullvec_fi_invariant": 0,   # times f_I stayed identical under (A)
        "band_fi_invariant": 0,      # times f_I stayed identical under (B)
        "trials": n,
    }

    for i in range(n):
        sid = int(rng.integers(0, 256))
        w_raw, meta = R.watermark(emb[i], sid, U, mu,
                                  e1_band, e0_band, e0_ppb, doc_id=i)

        # --- control: identity (no edit) ---
        v_ctrl = (w_raw - mu) @ U.T
        _, _, verdict = R.decode_e0(v_ctrl, R.MASTER_KEY, e0_band, e0_ppb)
        if verdict == "PASS":
            results["identity_pass"] += 1

        # fingerprint of the clean watermarked vector (baseline for f_I)
        f_i_clean = R.full_fingerprint(v_ctrl)

        # --- (A) full-vector permutation (multiset preserved exactly) ---
        v_pc = (w_raw - mu) @ U.T
        perm = rng.permutation(d)
        v_perm = v_pc[perm]
        if R.full_fingerprint(v_perm) == f_i_clean:
            results["fullvec_fi_invariant"] += 1
        _, _, verdict = R.decode_e0(v_perm, R.MASTER_KEY, e0_band, e0_ppb)
        if verdict == "REJECTED":
            results["fullvec_perm_reject"] += 1

        # --- (B) E_I-band-only permutation (multiset still preserved) ---
        v_pc = (w_raw - mu) @ U.T
        band_idx = np.arange(l2, h2)
        shuffled = rng.permutation(band_idx)
        v_band = v_pc.copy()
        v_band[band_idx] = v_pc[shuffled]
        if R.full_fingerprint(v_band) == f_i_clean:
            results["band_fi_invariant"] += 1
        _, _, verdict = R.decode_e0(v_band, R.MASTER_KEY, e0_band, e0_ppb)
        if verdict == "REJECTED":
            results["band_perm_reject"] += 1

    out = {
        "date": datetime.now().isoformat(),
        "experiment": "Remark 1 — swap-preserving (value-permuting) edits",
        "source": source,
        "d": d,
        "e0_band": [l2, h2],
        "e0_ppb": e0_ppb,
        **results,
        "identity_pass_rate": results["identity_pass"] / n,
        "fullvec_reject_rate": results["fullvec_perm_reject"] / n,
        "band_reject_rate": results["band_perm_reject"] / n,
    }
    return out


if __name__ == "__main__":
    out = run()
    print("=" * 64)
    print("Remark 1: swap-preserving (value-permuting) edits vs E_I")
    print("=" * 64)
    print(f"  source: {out['source']}   d={out['d']}   "
          f"E_I band={out['e0_band']}  ppb={out['e0_ppb']}")
    print()
    print(f"  [control] identity (no edit) PASS: "
          f"{out['identity_pass']}/{out['trials']} "
          f"({out['identity_pass_rate']:.1%})")
    print()
    print("  (A) full-vector random permutation (multiset preserved):")
    print(f"        f_I invariant:  {out['fullvec_fi_invariant']}/{out['trials']}"
          "   (expect 100% — multiset unchanged)")
    print(f"        E_I REJECT:     {out['fullvec_perm_reject']}/{out['trials']} "
          f"({out['fullvec_reject_rate']:.1%})   (expect ~100%)")
    print()
    print("  (B) E_I-band-only random permutation (multiset preserved):")
    print(f"        f_I invariant:  {out['band_fi_invariant']}/{out['trials']}"
          "   (expect 100% — values stay in band)")
    print(f"        E_I REJECT:     {out['band_perm_reject']}/{out['trials']} "
          f"({out['band_reject_rate']:.1%})   (expect ~100%)")
    print("=" * 64)
    print("Remark 1 holds iff: f_I invariant ~100% AND reject ~100%.")
    print("That is the point: multiset-invariance is available to the")
    print("attacker (f_I unchanged) yet confers no advantage (rejected).")
    print("=" * 64)

    os.makedirs("results", exist_ok=True)
    with open("results/swap_invariance.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/swap_invariance.json")
