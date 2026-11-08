#!/usr/bin/env python3
"""
Free-Position Permutation Attack (Walkthrough Sec. 6.1)
=======================================================
f_I hashes the SORTED multiset, so it is invariant to any rearrangement
of values. The confidence gate only covers PAIR POSITIONS:

    E_I keyed pairs :  8 bits x ppb x 2 positions   (secret)
    E_A sel pairs   :  8 bits x 3  x 2 = 48          (plaintext in meta)

Positions outside those pairs are covered only by the multiset. A
Level-1 verifier-side reviewer (reads vectors + metadata) can rearrange
the values on those free positions: multiset preserved -> f_I unchanged
-> same E_I pairs, untouched -> c_I ~ 1; sel untouched -> c_A ~ 1  =>
PASS, fully attributed. This script quantifies the retrieval impact and
confirms the hardened verifier (h_v + strict) rejects it.

Free set (identifiable without key or oracle):
  - tail [e0_hi, pc_dim)        : no pairs there
  - non-sel E_A band positions  : sel read from metadata
  - (variant B) head [0,e1_lo)  : reported separately; likely flips f_A

  (A) random permutation of the tail free set
  (B) random permutation of the full known free set
  (C) targeted assignment toward a query gradient:
        (i) promote the rank-6 candidate into top-5
        (ii) suppress the legitimate top-5
  (D) same attacks vs the hardened verifier -> expect 0 silent passes

Writes results/permutation_attack.json
Usage: python experiments/permutation_attack.py
"""
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
import ragmark_hardened as RH
from sklearn.decomposition import PCA

N_DOCS = 500
N_QUERIES = 500
KEY = R.MASTER_KEY
SEED = 2026


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def to_pc(v_raw, U, mu):
    return (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T


def to_raw(v_pc, U, mu):
    return v_pc @ U + mu


def free_positions(pc_dim, e1_band, e0_band, sel, include_head=False):
    """Positions covered by neither keyed E_I pairs nor stored sel pairs,
    identifiable without the key: tail beyond e0_hi, non-sel E_A band,
    optionally the head. E_I-band positions are excluded (keyed vs free
    there is unknown without an oracle)."""
    sel_pos = set()
    for bit_pairs in sel:
        for a, b in bit_pairs:
            sel_pos.add(a); sel_pos.add(b)
    free = list(range(e0_band[1], pc_dim))                     # tail
    free += [p for p in range(e1_band[0], e1_band[1])
             if p not in sel_pos]                              # non-sel E_A
    if include_head:
        free += list(range(0, e1_band[0]))
    return np.array(sorted(free), dtype=int)


def random_permutation(v_pc, free, rng):
    new = v_pc.copy()
    new[free] = v_pc[free][rng.permutation(len(free))]
    return new


def assignment_rearrange(v_pc, free, g, toward=True):
    """Rearrangement inequality: to maximize sum(new[free] * g[free])
    place the largest free value on the largest g coordinate; to
    minimize, reverse. 'toward' = pull the vector along g (increase
    similarity), else push away (decrease)."""
    new = v_pc.copy()
    vals = np.sort(v_pc[free])                 # ascending
    order = np.argsort(g[free])                # ascending g
    if toward:
        # largest value -> largest g
        new[free[order]] = vals
    else:
        # largest value -> smallest g
        new[free[order]] = vals[::-1]
    return new


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"date": datetime.now().isoformat(),
           "attack": "free-position permutation", "seed": SEED}

    print("=" * 62)
    print("Free-Position Permutation Attack (Sec. 6.1)")
    print("=" * 62)

    # ---- Setup -----------------------------------------------------
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    print(f"\n[SETUP] Loading corpus ({N_DOCS} docs)...")
    ds = load_dataset("rajpurkar/squad", split="train")
    seen, texts = set(), []
    for r in ds:
        if r["context"] not in seen and len(texts) < N_DOCS:
            seen.add(r["context"]); texts.append(r["context"])
    model = SentenceTransformer("all-mpnet-base-v2")
    E = np.asarray(model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=True))
    d = E.shape[1]
    pca = PCA(n_components=min(d, len(E))).fit(E)
    U, mu = pca.components_, pca.mean_
    pc_dim = U.shape[0]
    e1_band, e0_band, e0_ppb = R.compute_bands(d)
    print(f"  d={d}, pc_dim={pc_dim}, E_A={e1_band}, E_I={e0_band}, "
          f"ppb={e0_ppb}")

    print("  Watermarking (hardened: meta carries h_v)...")
    wm, metas = [], []
    for i in range(len(E)):
        w_raw, meta = RH.watermark_hardened(E[i], i % 8, U, mu,
                                            e1_band, e0_band, e0_ppb,
                                            doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta)
    wm = np.asarray(wm)
    wm_pc = np.asarray([to_pc(wm[i], U, mu) for i in range(len(wm))])

    base = sum(R.verify(wm[i], U, mu, e0_band, e0_ppb,
                        meta=metas[i], key=KEY)["verdict"] == "PASS"
               for i in range(len(wm)))
    print(f"  Baseline watermarked PASS: {base}/{len(wm)}")
    res["baseline_pass"] = f"{base}/{len(wm)}"

    # free-set size report (per-vector; sel differs by vector)
    fsz_tail = pc_dim - e0_band[1]
    fsz_full = np.mean([len(free_positions(pc_dim, e1_band, e0_band,
                                           metas[i]["sel"]))
                        for i in range(len(wm))])
    print(f"  free set: tail={fsz_tail}, full(mean)={fsz_full:.0f} "
          f"of {pc_dim}")
    res["free_tail"] = int(fsz_tail)
    res["free_full_mean"] = float(fsz_full)

    # ---- retrieval scaffold ---------------------------------------
    text2idx = {t: i for i, t in enumerate(texts)}
    qa = []
    for r in ds:
        if r["context"] in text2idx:
            qa.append((r["question"], text2idx[r["context"]]))
        if len(qa) >= 3000:
            break
    rng.shuffle(qa)
    qa = qa[:N_QUERIES]
    Q = np.asarray(model.encode([q for q, _ in qa],
                                normalize_embeddings=True))
    gt = np.array([i for _, i in qa])
    s_wm = unit(Q) @ unit(wm).T
    rank_wm = np.argsort(-s_wm, axis=1)

    def verify_pass(a_raw, i, hardened=False):
        if hardened:
            v = RH.verify_hardened(a_raw, U, mu, e0_band, e0_ppb,
                                   meta=metas[i], key=KEY,
                                   expected_doc_id=metas[i]["doc_id"])
        else:
            v = R.verify(a_raw, U, mu, e0_band, e0_ppb,
                         meta=metas[i], key=KEY)
        return v["verdict"]

    # =================================================================
    # (A) tail-only random permutation
    # =================================================================
    print("\n[A] Untargeted random permutation (tail free set)")
    _run_untargeted(wm, wm_pc, metas, U, mu, e0_band, e0_ppb,
                    Q, gt, s_wm, rank_wm, pc_dim, e1_band,
                    include_head=False, tail_only=True,
                    rng=rng, res=res, tag="A_tail", verify_pass=verify_pass)

    # =================================================================
    # (B) full known free set random permutation
    # =================================================================
    print("\n[B] Untargeted random permutation (full known free set)")
    _run_untargeted(wm, wm_pc, metas, U, mu, e0_band, e0_ppb,
                    Q, gt, s_wm, rank_wm, pc_dim, e1_band,
                    include_head=False, tail_only=False,
                    rng=rng, res=res, tag="B_full", verify_pass=verify_pass)

    # =================================================================
    # (C) targeted assignment: promotion + suppression
    # =================================================================
    print("\n[C] Targeted assignment (promote rank-6 / suppress top-5)")
    promo_pass = promo_succ = 0
    supp_slots = []
    supp_pass = 0
    supp_queries = 0
    for qi in range(len(Q)):
        g = U @ unit(Q[qi])
        # (i) promotion: take the rank-6 doc (0-indexed rank 5), pull it
        cand = int(rank_wm[qi, 5])
        free = free_positions(pc_dim, e1_band, e0_band, metas[cand]["sel"])
        a_pc = assignment_rearrange(wm_pc[cand], free, g, toward=True)
        a_raw = to_raw(a_pc, U, mu)
        promo_pass += (verify_pass(a_raw, cand) == "PASS")
        s_after = float(unit(Q[qi]) @ unit(a_raw))
        col = s_wm[qi].copy(); col[cand] = s_after
        new_rank = int(np.sum(col > col[cand]))
        promo_succ += (new_rank < 5)
        # (ii) suppression: push each legit top-5 doc AWAY from the query
        opened = 0
        for slot in range(5):
            di = int(rank_wm[qi, slot])
            free_d = free_positions(pc_dim, e1_band, e0_band,
                                    metas[di]["sel"])
            a_pc = assignment_rearrange(wm_pc[di], free_d, g, toward=False)
            a_raw = to_raw(a_pc, U, mu)
            supp_pass += (verify_pass(a_raw, di) == "PASS")
            s_after = float(unit(Q[qi]) @ unit(a_raw))
            col = s_wm[qi].copy(); col[di] = s_after
            if int(np.sum(col > col[di])) >= 5:      # fell out of top-5
                opened += 1
        supp_slots.append(opened)
        supp_queries += (opened >= 1)
    supp_slots = np.array(supp_slots)
    print(f"  Promotion PASS (silent):    {promo_pass}/{len(Q)}")
    print(f"  Promotion into top-5:       {promo_succ}/{len(Q)} "
          f"({100*promo_succ/len(Q):.2f}%)")
    print(f"  Suppression PASS (silent):  {supp_pass}/{5*len(Q)}")
    print(f"  Slots opened/query:         mean={supp_slots.mean():.3f} "
          f"max={supp_slots.max()}")
    print(f"  Queries with >=1 opened:    {supp_queries}/{len(Q)} "
          f"({100*supp_queries/len(Q):.2f}%)")
    res["promotion_pass"] = f"{promo_pass}/{len(Q)}"
    res["promotion_into_top5"] = f"{promo_succ}/{len(Q)}"
    res["promotion_pct"] = float(100*promo_succ/len(Q))
    res["suppression_slots_mean"] = float(supp_slots.mean())
    res["suppression_queries_pct"] = float(100*supp_queries/len(Q))

    # =================================================================
    # (D) hardened verifier
    # =================================================================
    print("\n[D] Same attacks vs HARDENED verifier (h_v + strict)")
    h_pass = 0
    for qi in range(min(len(Q), 200)):
        g = U @ unit(Q[qi])
        cand = int(rank_wm[qi, 5])
        free = free_positions(pc_dim, e1_band, e0_band, metas[cand]["sel"])
        a_pc = assignment_rearrange(wm_pc[cand], free, g, toward=True)
        a_raw = to_raw(a_pc, U, mu)
        h_pass += (verify_pass(a_raw, cand, hardened=True) == "PASS")
    # also: permute + delete metadata (strict mode)
    del_pass = 0
    for i in range(min(len(wm), 200)):
        free = free_positions(pc_dim, e1_band, e0_band, metas[i]["sel"])
        a_pc = random_permutation(wm_pc[i], free, rng)
        a_raw = to_raw(a_pc, U, mu)
        v = RH.verify_hardened(a_raw, U, mu, e0_band, e0_ppb,
                               meta=None, key=KEY, strict=True)
        del_pass += (v["verdict"] == "PASS")
    print(f"  Permute, meta present, hardened PASS:  {h_pass}/200  "
          f"(expect 0)")
    print(f"  Permute + delete meta, strict PASS:    {del_pass}/200  "
          f"(expect 0)")
    res["hardened_permute_pass"] = f"{h_pass}/200"
    res["hardened_permute_delete_pass"] = f"{del_pass}/200"

    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  free set (of {pc_dim}): tail {fsz_tail}, full {fsz_full:.0f}")
    print(f"  targeted promotion into top-5: {res['promotion_into_top5']} "
          f"({res['promotion_pct']:.1f}%)")
    print(f"  suppression: {res['suppression_queries_pct']:.1f}% of queries")
    print(f"  hardened: {h_pass}/200 and {del_pass}/200 (both expect 0)")
    print("=" * 62)
    with open("results/permutation_attack.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/permutation_attack.json")


def _run_untargeted(wm, wm_pc, metas, U, mu, e0_band, e0_ppb,
                    Q, gt, s_wm, rank_wm, pc_dim, e1_band,
                    include_head, tail_only, rng, res, tag, verify_pass):
    npass = nfp = 0
    l2s, coss = [], []
    adv = np.empty_like(wm)
    for i in range(len(wm)):
        if tail_only:
            free = np.arange(e0_band[1], pc_dim)
        else:
            free = free_positions(pc_dim, e1_band, e0_band,
                                  metas[i]["sel"], include_head)
        a_pc = random_permutation(wm_pc[i], free, rng)
        a_raw = to_raw(a_pc, U, mu)
        adv[i] = a_raw
        npass += (verify_pass(a_raw, i) == "PASS")
        nfp += (R.full_fingerprint(a_pc) == metas[i]["f_i"])
        l2s.append(np.linalg.norm(a_raw - wm[i]))
        coss.append(float(unit(a_raw) @ unit(wm[i])))
    print(f"  Attacked PASS (silent):  {npass}/{len(wm)}")
    print(f"  f_I invariant:           {nfp}/{len(wm)}")
    print(f"  L2 drift:  mean={np.mean(l2s):.4f} max={np.max(l2s):.4f}")
    print(f"  cos(wm,adv): mean={np.mean(coss):.5f} min={np.min(coss):.5f}")
    s_adv = unit(Q) @ unit(adv).T
    rank_adv = np.argsort(-s_adv, axis=1)
    for k in (1, 5, 10):
        ov = np.mean([len(set(rank_wm[i, :k]) & set(rank_adv[i, :k])) /
                      len(set(rank_wm[i, :k]) | set(rank_adv[i, :k]))
                      for i in range(len(Q))])
        print(f"  Top-{k} overlap (wm vs adv): {ov:.3f}")
        res[f"{tag}_top{k}_overlap"] = float(ov)
    for k in (5, 10):
        rw = np.mean([gt[i] in rank_wm[i, :k] for i in range(len(gt))])
        ra = np.mean([gt[i] in rank_adv[i, :k] for i in range(len(gt))])
        print(f"  GT Recall@{k}: wm={rw:.3f} adv={ra:.3f} diff={ra-rw:+.3f}")
        res[f"{tag}_recall{k}_wm"] = float(rw)
        res[f"{tag}_recall{k}_adv"] = float(ra)
    res[f"{tag}_pass"] = f"{npass}/{len(wm)}"
    res[f"{tag}_fI_invariant"] = f"{nfp}/{len(wm)}"
    res[f"{tag}_l2_mean"] = float(np.mean(l2s))
    res[f"{tag}_l2_max"] = float(np.max(l2s))
    res[f"{tag}_cos_mean"] = float(np.mean(coss))


if __name__ == "__main__":
    main()
