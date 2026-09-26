#!/usr/bin/env python3
"""
Experiment 2: Metadata-Replay / Cross-Source-Swap
===================================================
Tests whether the dual-layer s_A == s_I check is load-bearing.

Attack: take source-A's valid HMAC-signed metadata (fingerprint f_A +
selection indices, encoding source ID A) and attach it to source-B's
valid watermarked vector (whose E_I encodes source ID B). The HMAC
still verifies (metadata is genuine, just reattached). We then ask:

  - E_A-only verification: decodes A from the stored pairs -> ACCEPTS
    (this is the vulnerability: wrong provenance, undetected)
  - Dual-layer verification: E_I decodes B from the vector; if
    c_I >= tau AND c_A >= tau but s_A (=A) != s_I (=B) -> REJECT

We measure, for every cross-source swap:
  1. E_A-only verdict (expect: ACCEPT, decoding the WRONG source)
  2. Dual-layer verdict, and WHICH gate rejected:
       - "c_I<tau"   : integrity confidence gate fired first
       - "c_A<tau"   : attribution confidence gate fired
       - "s_A!=s_I"  : the ID-mismatch cross-check fired (the case
                       that proves the check is independently useful)
       - "ACCEPT"    : slipped through (should be ~0)

This gives the INDEPENDENT firing rate of the s_A==s_I check:
how often it is the *sole* reason a swap is caught.

Domain-separated seeds (matches Experiment 1 / the paper spec).

Usage: python experiments/exp2_metadata_replay.py
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
TAG_ATTR = "attr"
TAG_INTEG = "integ"


def compute_bands(d):
    e1_lo = max(1, int(round(0.04 * d)))
    e1_hi = int(round(0.2 * d))
    e0_lo = e1_hi
    e0_hi = int(round(0.5 * d))
    e0_ppb = min(10, (e0_hi - e0_lo) // (2 * N_BITS))
    return (e1_lo, e1_hi), (e0_lo, e0_hi), e0_ppb


def derive_pairs_tagged(key, tag, fp, band, ppb, n_bits=N_BITS):
    seed = int(hashlib.sha256(
        (key + "|" + tag + "|" + fp).encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    idx = list(range(band[0], band[1]))
    rng.shuffle(idx)
    pairs, pos = [], 0
    for b in range(n_bits):
        bp = []
        for _ in range(ppb):
            bp.append((idx[pos], idx[pos + 1])); pos += 2
        pairs.append(bp)
    return pairs


def greedy_select_e1(v_pc, sid, fp, key, band,
                     tag=TAG_ATTR, ppb=PPB_E1, n_bits=N_BITS, delta=DELTA):
    w = v_pc.copy()
    seed = int(hashlib.sha256(
        (key + "|" + tag + "|" + fp).encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    allidx = list(range(band[0], band[1])); rng.shuffle(allidx)
    committed, sel = set(), []
    for b in range(n_bits):
        bv = (sid >> b) & 1
        avail = [i for i in allidx if i not in committed]
        cands = []
        for ci in range(len(avail) // 2):
            a, b_ = avail[2 * ci], avail[2 * ci + 1]
            gap = (w[a] - w[b_]) if bv == 1 else (w[b_] - w[a])
            cands.append((gap, a, b_))
        cands.sort(key=lambda x: -x[0])
        chosen = []
        for gap, a, b_ in cands[:ppb]:
            committed.add(a); committed.add(b_); chosen.append((a, b_))
            if gap < 0:
                mid = (w[a] + w[b_]) / 2
                if bv == 1:
                    w[a], w[b_] = mid + delta, mid - delta
                else:
                    w[a], w[b_] = mid - delta, mid + delta
        sel.append(chosen)
    return w, sel


def decode_e1(v_pc, sel, ppb=PPB_E1, n_bits=N_BITS):
    bits, confs = [], []
    for b in range(n_bits):
        votes = [1 if v_pc[a] > v_pc[b_] else 0
                 for a, b_ in sel[b] if v_pc[a] != v_pc[b_]]
        if not votes:
            bits.append(0); confs.append(0.0)
        else:
            ones = sum(votes)
            bits.append(1 if ones > len(votes) / 2 else 0)
            confs.append(max(ones, len(votes) - ones) / ppb)
    return sum(bit << j for j, bit in enumerate(bits)), float(np.mean(confs))


def full_fp(v_pc):
    c = np.round(v_pc, 2) + 0.0
    return hashlib.sha256(str(tuple(np.sort(c))).encode()).hexdigest()[:16]


def e0_embed(v_pc, sid, key, band, ppb, tag=TAG_INTEG, n_bits=N_BITS):
    w = v_pc.copy()
    pairs = derive_pairs_tagged(key, tag, full_fp(v_pc), band, ppb, n_bits)
    for b in range(n_bits):
        bv = (sid >> b) & 1
        for a, b_ in pairs[b]:
            if bv == 1 and w[a] < w[b_]:
                w[a], w[b_] = w[b_], w[a]
            elif bv == 0 and w[a] > w[b_]:
                w[a], w[b_] = w[b_], w[a]
    return w


def e0_decode(v_pc, key, band, ppb, tag=TAG_INTEG, n_bits=N_BITS):
    pairs = derive_pairs_tagged(key, tag, full_fp(v_pc), band, ppb, n_bits)
    bits, confs = [], []
    for b in range(n_bits):
        votes = [1 if v_pc[a] > v_pc[b_] else 0
                 for a, b_ in pairs[b] if v_pc[a] != v_pc[b_]]
        if not votes:
            bits.append(0); confs.append(0.0)
        else:
            ones = sum(votes)
            bits.append(1 if ones > len(votes) / 2 else 0)
            confs.append(max(ones, len(votes) - ones) / ppb)
    return sum(bit << j for j, bit in enumerate(bits)), float(np.mean(confs))


def embed_full(v_raw, sid, U, mu, key, e1b, e0b, e0ppb):
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T
    e1_fp = R.content_fingerprint(v_raw)
    v_pc, sel = greedy_select_e1(v_pc, sid, e1_fp, key, e1b)
    v_pc = e0_embed(v_pc, sid, key, e0b, e0ppb)
    return v_pc @ U + mu, {"f_A": e1_fp, "sel": sel, "s": sid}


# =====================================================================
# Verification variants
# =====================================================================
def verify_ea_only(v_raw, U, mu, meta, tau=TAU):
    """E_A-only: decode using stored sel. Returns (sid, conf, verdict)."""
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T
    sid, conf = decode_e1(v_pc, meta["sel"])
    return sid, conf, ("PASS" if conf >= tau else "REJECTED")


def verify_dual(v_raw, U, mu, key, e0b, e0ppb, meta, tau=TAU):
    """
    Dual-layer: E_I first, then E_A, then s_A == s_I.
    Returns (sid, verdict, reason).
    """
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T
    s_I, c_I = e0_decode(v_pc, key, e0b, e0ppb)
    if c_I < tau:
        return None, "REJECTED", "c_I<tau"
    s_A, c_A = decode_e1(v_pc, meta["sel"])
    if c_A < tau:
        return None, "REJECTED", "c_A<tau"
    if s_A != s_I:
        return None, "REJECTED", "s_A!=s_I"
    return s_A, "PASS", "accept"


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(2026)
    res = {"date": datetime.now().isoformat()}

    print("=" * 64)
    print("Experiment 2: Metadata-Replay / Cross-Source-Swap")
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

    # Watermark with DISTINCT source IDs so swaps are cross-source.
    # Use source_id = i % 8, so many vectors share each ID; we only
    # swap metadata between vectors whose IDs DIFFER.
    wm, metas, sids = [], [], []
    for i in range(len(E)):
        sid = i % 8
        w, m = embed_full(E[i], sid, U, mu, KEY, e1b, e0b, e0ppb)
        wm.append(w); metas.append(m); sids.append(sid)
    wm = np.array(wm)

    # Sanity: legitimate vectors pass dual-layer
    ok = sum(verify_dual(wm[i], U, mu, KEY, e0b, e0ppb, metas[i])[1] == "PASS"
             for i in range(len(wm)))
    print(f"  legitimate dual-layer pass: {ok}/500")
    res["legit_pass"] = ok

    # ================================================================
    # CROSS-SOURCE SWAP
    # ================================================================
    print("\n[ATTACK] Cross-source metadata swap")
    print("  For each vector B, attach metadata from a vector A with a")
    print("  DIFFERENT source ID. Verify B with A's metadata.")

    n_trials = 500
    ea_accept = 0            # E_A-only wrongly accepts
    ea_wrong_source = 0      # E_A-only accepts AND decodes A's id (not B's)
    reasons = {"c_I<tau": 0, "c_A<tau": 0, "s_A!=s_I": 0, "accept": 0}

    trials_done = 0
    attempts = 0
    while trials_done < n_trials and attempts < n_trials * 20:
        attempts += 1
        b_idx = rng.integers(0, len(wm))
        a_idx = rng.integers(0, len(wm))
        if sids[a_idx] == sids[b_idx]:
            continue  # need cross-source
        trials_done += 1

        v_B = wm[b_idx]          # source-B vector
        meta_A = metas[a_idx]    # source-A metadata (valid HMAC, different s)

        # E_A-only verification with A's metadata
        s_ea, c_ea, verd_ea = verify_ea_only(v_B, U, mu, meta_A)
        if verd_ea == "PASS":
            ea_accept += 1
            if s_ea == meta_A["s"] and meta_A["s"] != sids[b_idx]:
                ea_wrong_source += 1

        # Dual-layer verification with A's metadata
        _, verd_d, reason = verify_dual(v_B, U, mu, KEY, e0b, e0ppb, meta_A)
        reasons[reason] += 1

    print(f"\n  trials: {trials_done}")
    print(f"\n  --- E_A-only (single-layer) ---")
    print(f"    accepted (wrong provenance): {ea_accept}/{trials_done}")
    print(f"    accepted AND decoded A's ID: {ea_wrong_source}/{trials_done}")
    print(f"\n  --- Dual-layer: rejection reason breakdown ---")
    print(f"    c_I < tau  (integrity gate):   {reasons['c_I<tau']}")
    print(f"    c_A < tau  (attribution gate): {reasons['c_A<tau']}")
    print(f"    s_A != s_I (ID cross-check):   {reasons['s_A!=s_I']}  "
          f"<-- independent firing")
    print(f"    ACCEPT (slipped through):      {reasons['accept']}")

    total_caught = trials_done - reasons["accept"]
    print(f"\n  dual-layer caught: {total_caught}/{trials_done}")
    if total_caught > 0:
        indep = reasons["s_A!=s_I"]
        print(f"  s_A!=s_I was the SOLE reason in {indep}/{total_caught} "
              f"({100*indep/total_caught:.1f}%) of catches")

    res["ea_accept"] = ea_accept
    res["ea_wrong_source"] = ea_wrong_source
    res["dual_reasons"] = reasons
    res["dual_caught"] = total_caught
    res["trials"] = trials_done

    # ================================================================
    # SAME-SOURCE SWAP (control — cross-check CANNOT catch these)
    # ================================================================
    print("\n[CONTROL] Same-source metadata swap")
    print("  Attach metadata from a DIFFERENT vector with the SAME source")
    print("  ID. s_A == s_I by construction, so the ID check cannot fire;")
    print("  only c_A<tau (wrong sel positions) can catch these.")

    ss_reasons = {"c_I<tau": 0, "c_A<tau": 0, "s_A!=s_I": 0, "accept": 0}
    ss_done, ss_att = 0, 0
    while ss_done < 200 and ss_att < 200 * 40:
        ss_att += 1
        b_idx = rng.integers(0, len(wm))
        a_idx = rng.integers(0, len(wm))
        if a_idx == b_idx or sids[a_idx] != sids[b_idx]:
            continue
        ss_done += 1
        _, _, reason = None, None, verify_dual(
            wm[b_idx], U, mu, KEY, e0b, e0ppb, metas[a_idx])[2]
        ss_reasons[reason] += 1

    print(f"\n  trials: {ss_done}")
    print(f"    c_I < tau:   {ss_reasons['c_I<tau']}")
    print(f"    c_A < tau:   {ss_reasons['c_A<tau']}")
    print(f"    s_A != s_I:  {ss_reasons['s_A!=s_I']}  (expect 0 — same source)")
    print(f"    ACCEPT:      {ss_reasons['accept']}  <-- same-source gap")
    res["same_source_reasons"] = ss_reasons
    res["same_source_trials"] = ss_done

    print(f"\n{'='*64}")
    print("INTERPRETATION:")
    print("  - ea_wrong_source > 0 proves E_A-only accepts wrong provenance.")
    print("  - s_A!=s_I count is the cross-check's INDEPENDENT value.")
    print("  - same-source ACCEPT count is the documented scope limit")
    print("    (cross-check catches cross-source swaps only).")
    print(f"{'='*64}")
    with open("results/metadata_replay.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/metadata_replay.json")


if __name__ == "__main__":
    main()
