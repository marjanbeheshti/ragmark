#!/usr/bin/env python3
"""
Sub-Resolution (Within-Bucket) Adversarial Attack
=================================================
Reviewer Issue 2 (the third modification class neither Prop. 1 nor
Remark 1 covers): a perturbation confined to the 2-decimal rounding
buckets leaves the rounded multiset -- and thus f_I -- invariant. If it
additionally preserves every keyed pair ordering (E_A stored + E_I
derived), it passes verify() with certainty. This is a genuine silent
budget. This script quantifies its WORST-CASE retrieval impact, to test
whether the dead zone is operationally exploitable.

  (A) Untargeted max-drift: push every PC coordinate of every document
      to the far edge of its bucket (largest geometric drift the budget
      allows), then measure retrieval degradation vs the watermarked
      baseline (Spearman, top-k overlap, GT recall, rank shift).

  (B) Targeted injection: for a target query, push a poison document
      toward the query WITHIN buckets (max keyed budget along the
      similarity gradient U @ q), then measure (i) rank improvement of an
      already-relevant doc, and (ii) whether a non-relevant doc can be
      injected into top-5.

Every attacked vector is passed through ragmark.verify() to confirm it is
ACCEPTED (silent) and that f_I is unchanged -- i.e. the attack really does
live in the dead zone. If passes < 100%, the perturbation model is too
aggressive and MARGIN should be raised.

Design note: E_A and E_I bands are disjoint and pairs within a layer are
disjoint, so every coordinate belongs to at most ONE keyed pair. Order
preservation therefore clips each pair independently (no coupling).

Writes results/sub_resolution.json

Usage: python experiments/sub_resolution_attack.py
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
from sklearn.decomposition import PCA


# =====================================================================
# Configuration
# =====================================================================
N_DOCS = 500
KEY = R.MASTER_KEY
BUCKET = 0.005          # half-width of a 2-dp rounding bucket
MARGIN = 1e-4           # stay off the rounding boundary AND keep strict
                        # pair orderings (<< BUCKET, negligible budget loss)


# =====================================================================
# Attack primitives
# =====================================================================
def bucket_edges(v_pc):
    """Per-coordinate allowed interval that keeps round(.,2) unchanged
    (hence f_I invariant), held MARGIN off the boundary."""
    r = np.round(v_pc, 2)
    lo = r - BUCKET + MARGIN
    hi = r + BUCKET - MARGIN
    # never clip the original outside its own (numerical) bucket
    lo = np.minimum(lo, v_pc)
    hi = np.maximum(hi, v_pc)
    return lo, hi


def all_keyed_pairs(v_pc, meta, e0_band, e0_ppb, key=KEY):
    """Every keyed pair the verifier will read: E_A stored selection +
    E_I fingerprint-derived pairs. Each coordinate appears in <=1 pair."""
    pairs = []
    for bit_pairs in meta["sel"]:            # E_A (stored)
        pairs.extend((a, b) for a, b in bit_pairs)
    f_i = R.full_fingerprint(v_pc)           # E_I (self-derived)
    for bit_pairs in R.derive_pairs(key, R.TAG_INTEG, f_i, e0_band, e0_ppb):
        pairs.extend((a, b) for a, b in bit_pairs)
    return pairs


def within_bucket_attack(v_pc, direction, meta, e0_band, e0_ppb, key=KEY):
    """Move each coordinate to its bucket edge in `direction`
    (+1 up / -1 down / 0 hold), then clip to preserve every keyed pair
    ordering. Returns the attacked PC vector (f_I invariant, verify-safe).
    `direction` is the attacker's objective gradient sign per coordinate.
    """
    lo, hi = bucket_edges(v_pc)
    new = v_pc.copy()
    up = direction > 0
    dn = direction < 0
    new[up] = hi[up]
    new[dn] = lo[dn]

    # Preserve pair orderings (each coord in <=1 pair -> independent clips)
    for a, b in all_keyed_pairs(v_pc, meta, e0_band, e0_ppb, key):
        # identify which of (a,b) must stay the larger, from the ORIGINAL
        if v_pc[a] >= v_pc[b]:
            hc, lc = a, b
        else:
            hc, lc = b, a
        gap = new[hc] - new[lc]
        if gap >= MARGIN:
            continue
        # project onto {new[hc] - new[lc] >= MARGIN} within buckets
        need = (MARGIN - gap) / 2.0
        nh = min(hi[hc], new[hc] + need)
        nl = max(lo[lc], new[lc] - need)
        if nh - nl >= MARGIN:
            new[hc], new[lc] = nh, nl
        else:
            # cannot satisfy inside buckets -> freeze this pair (rare)
            new[hc], new[lc] = v_pc[hc], v_pc[lc]
    return new


def to_raw(v_pc, U, mu):
    return v_pc @ U + mu


def to_pc(v_raw, U, mu):
    return (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


# =====================================================================
# Main
# =====================================================================
def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(2026)
    results = {"date": datetime.now().isoformat(),
               "attack": "sub-resolution within-bucket",
               "bucket_halfwidth": BUCKET, "margin": MARGIN}

    print("=" * 62)
    print("Sub-Resolution (Within-Bucket) Adversarial Attack")
    print("=" * 62)

    # ---- Setup (mirrors supplementary.py) --------------------------
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
    e1_band, e0_band, e0_ppb = R.compute_bands(d)
    print(f"  d={d}, E_A={e1_band}, E_I={e0_band}, ppb={e0_ppb}")

    # ---- Watermark (keep metadata) ---------------------------------
    print("  Watermarking...")
    wm, metas = [], []
    for i in range(len(E)):
        w_raw, meta = R.watermark(E[i], i % 8, U, mu,
                                  e1_band, e0_band, e0_ppb,
                                  doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta)
    wm = np.asarray(wm)
    wm_pc = np.asarray([to_pc(wm[i], U, mu) for i in range(len(wm))])

    # baseline sanity: watermarked vectors pass
    base_pass = sum(R.verify(wm[i], U, mu, e0_band, e0_ppb,
                             meta=metas[i], key=KEY)["verdict"] == "PASS"
                    for i in range(len(wm)))
    print(f"  Baseline watermarked PASS: {base_pass}/{len(wm)}")
    results["baseline_pass"] = f"{base_pass}/{len(wm)}"

    # =================================================================
    # (A) UNTARGETED MAX-DRIFT
    # =================================================================
    print("\n[A] Untargeted max-drift (every coord to far bucket edge)")
    adv_pc = np.empty_like(wm_pc)
    for i in range(len(wm_pc)):
        # direction = toward the FARTHER bucket edge -> maximize |drift|
        lo, hi = bucket_edges(wm_pc[i])
        far_up = (hi - wm_pc[i]) >= (wm_pc[i] - lo)
        direction = np.where(far_up, 1.0, -1.0)
        adv_pc[i] = within_bucket_attack(wm_pc[i], direction, metas[i],
                                         e0_band, e0_ppb, KEY)
    adv = np.asarray([to_raw(adv_pc[i], U, mu) for i in range(len(adv_pc))])

    # verify: the attacked vectors must PASS (silent) and keep f_I
    n_pass = n_fp = 0
    for i in range(len(adv)):
        v = R.verify(adv[i], U, mu, e0_band, e0_ppb, meta=metas[i], key=KEY)
        n_pass += (v["verdict"] == "PASS")
        n_fp += (R.full_fingerprint(adv_pc[i]) == metas[i]["f_i"])
    print(f"  Attacked PASS (silent):   {n_pass}/{len(adv)}")
    print(f"  f_I invariant:            {n_fp}/{len(adv)}")
    results["untargeted_pass"] = f"{n_pass}/{len(adv)}"
    results["untargeted_fI_invariant"] = f"{n_fp}/{len(adv)}"

    # geometry of the achieved perturbation (raw space, unit vectors)
    l2 = np.linalg.norm(adv - wm, axis=1)
    cos_wm_adv = np.sum(unit(adv) * unit(wm), axis=1)
    print(f"  L2 drift raw:   mean={l2.mean():.4f} max={l2.max():.4f}")
    print(f"  cos(wm, adv):   mean={cos_wm_adv.mean():.5f} "
          f"min={cos_wm_adv.min():.5f}")
    results["untargeted_l2_mean"] = float(l2.mean())
    results["untargeted_l2_max"] = float(l2.max())
    results["untargeted_cos_mean"] = float(cos_wm_adv.mean())
    results["untargeted_cos_min"] = float(cos_wm_adv.min())

    # ---- retrieval impact: watermarked baseline vs attacked DB -----
    text2idx = {t: i for i, t in enumerate(texts)}
    qa = []
    for r in ds:
        if r["context"] in text2idx:
            qa.append((r["question"], text2idx[r["context"]]))
        if len(qa) >= 2500:
            break
    rng.shuffle(qa)
    qa = qa[:500]
    Q = np.asarray(model.encode([q for q, _ in qa],
                                normalize_embeddings=True))
    gt_idx = np.array([i for _, i in qa])

    def sims(db, queries):
        return unit(queries) @ unit(db).T

    s_wm, s_adv = sims(wm, Q), sims(adv, Q)
    rank_wm = np.argsort(-s_wm, axis=1)
    rank_adv = np.argsort(-s_adv, axis=1)

    rhos = [scipy_stats.spearmanr(s_wm[i], s_adv[i])[0] for i in range(len(Q))]
    print(f"  Spearman(wm, adv rankings): {np.mean(rhos):.4f}")
    results["untargeted_spearman"] = float(np.mean(rhos))

    for k in (1, 5, 10):
        ov = np.mean([len(set(rank_wm[i, :k]) & set(rank_adv[i, :k])) /
                      len(set(rank_wm[i, :k]) | set(rank_adv[i, :k]))
                      for i in range(len(Q))])
        print(f"  Top-{k} overlap (wm vs adv): {ov:.3f}")
        results[f"untargeted_top{k}_overlap"] = float(ov)

    for k in (1, 5, 10):
        rw = np.mean([gt_idx[i] in rank_wm[i, :k] for i in range(len(gt_idx))])
        ra = np.mean([gt_idx[i] in rank_adv[i, :k] for i in range(len(gt_idx))])
        print(f"  GT Recall@{k}: wm={rw:.3f} adv={ra:.3f} diff={ra-rw:+.3f}")
        results[f"untargeted_recall{k}_wm"] = float(rw)
        results[f"untargeted_recall{k}_adv"] = float(ra)

    # GT-doc rank shift under the attack
    def gt_rank(rankmat):
        out = []
        for i in range(len(gt_idx)):
            p = np.where(rankmat[i] == gt_idx[i])[0]
            out.append(p[0] if len(p) else 999)
        return np.array(out, dtype=float)
    shift = gt_rank(rank_adv) - gt_rank(rank_wm)
    unchanged = int(np.sum(shift == 0))
    print(f"  GT rank shift: mean={shift.mean():+.2f} "
          f"max|.|={np.abs(shift).max():.0f}  unchanged="
          f"{unchanged}/{len(shift)} ({100*unchanged/len(shift):.1f}%)")
    results["untargeted_rank_shift_mean"] = float(shift.mean())
    results["untargeted_rank_shift_maxabs"] = float(np.abs(shift).max())
    results["untargeted_docs_unchanged_pct"] = float(100*unchanged/len(shift))

    # =================================================================
    # (B) TARGETED INJECTION
    # =================================================================
    print("\n[B] Targeted injection (push poison toward a target query)")

    def targeted_adv(doc_i, q_vec, toward=1.0):
        """Attack doc_i's watermarked vector along the similarity gradient
        toward (+1) or away from (-1) q_vec, within buckets."""
        g = U @ unit(q_vec)                      # per-PC similarity gradient
        direction = toward * np.sign(g)
        a_pc = within_bucket_attack(wm_pc[doc_i], direction, metas[doc_i],
                                    e0_band, e0_ppb, KEY)
        return a_pc, to_raw(a_pc, U, mu)

    # (i) boost an ALREADY-RELEVANT doc: target = its own GT query
    gains, rank_impr, tpass = [], [], 0
    for qi in range(len(Q)):
        di = gt_idx[qi]
        a_pc, a_raw = targeted_adv(di, Q[qi])
        tpass += (R.verify(a_raw, U, mu, e0_band, e0_ppb,
                           meta=metas[di], key=KEY)["verdict"] == "PASS")
        s_before = float(unit(Q[qi]) @ unit(wm[di]))
        s_after = float(unit(Q[qi]) @ unit(a_raw))
        gains.append(s_after - s_before)
        # rank of di for qi, others held at watermarked
        col = s_wm[qi].copy()
        col[di] = s_after
        r_before = int(np.where(rank_wm[qi] == di)[0][0])
        r_after = int(np.sum(col > col[di]))    # 0-indexed rank
        rank_impr.append(r_before - r_after)    # positive = moved up
    gains = np.array(gains); rank_impr = np.array(rank_impr)
    print(f"  Attacked PASS (silent):     {tpass}/{len(Q)}")
    print(f"  Sim gain toward GT query:   mean={gains.mean():+.5f} "
          f"max={gains.max():+.5f}")
    print(f"  GT-doc rank improvement:    mean={rank_impr.mean():+.3f} "
          f"max={rank_impr.max():+d} (positions moved up)")
    results["targeted_relevant_pass"] = f"{tpass}/{len(Q)}"
    results["targeted_sim_gain_mean"] = float(gains.mean())
    results["targeted_sim_gain_max"] = float(gains.max())
    results["targeted_rank_improve_mean"] = float(rank_impr.mean())
    results["targeted_rank_improve_max"] = int(rank_impr.max())

    # (ii) INJECT a non-relevant doc into top-5 that was outside it
    inj_attempts = inj_success = 0
    for qi in range(len(Q)):
        outside = [j for j in rank_wm[qi, 20:] ]      # clearly non-top docs
        if not outside:
            continue
        di = int(rng.choice(outside))
        inj_attempts += 1
        a_pc, a_raw = targeted_adv(di, Q[qi])
        s_after = float(unit(Q[qi]) @ unit(a_raw))
        col = s_wm[qi].copy(); col[di] = s_after
        new_rank = int(np.sum(col > col[di]))         # 0-indexed
        inj_success += (new_rank < 5)
    print(f"  Injection into top-5 (random cand): {inj_success}/{inj_attempts} "
          f"({100*inj_success/max(inj_attempts,1):.2f}%)")
    results["injection_success"] = f"{inj_success}/{inj_attempts}"
    results["injection_success_pct"] = float(
        100*inj_success/max(inj_attempts, 1))

    # (iii) BEST-POSITIONED injection: realistic Level-1 attacker picks the
    #       doc just outside top-5 (rank 6) and nudges it in. Worst case.
    bp_success = 0
    bp_rank_impr = []
    for qi in range(len(Q)):
        cand = int(rank_wm[qi, 5])             # 6th-ranked doc for this query
        _, a_raw = targeted_adv(cand, Q[qi])
        s_after = float(unit(Q[qi]) @ unit(a_raw))
        col = s_wm[qi].copy(); col[cand] = s_after
        new_rank = int(np.sum(col > col[cand]))    # 0-indexed
        bp_success += (new_rank < 5)
        bp_rank_impr.append(5 - new_rank)          # positions gained past #6
    bp_rank_impr = np.array(bp_rank_impr)
    print(f"  Best-positioned (6th) injection top-5: {bp_success}/{len(Q)} "
          f"({100*bp_success/len(Q):.2f}%)  "
          f"rank gain mean={bp_rank_impr.mean():+.3f} max={bp_rank_impr.max():+d}")
    results["injection_bestpos"] = f"{bp_success}/{len(Q)}"
    results["injection_bestpos_pct"] = float(100*bp_success/len(Q))
    results["injection_bestpos_rankgain_mean"] = float(bp_rank_impr.mean())

    # (iv) SUPPRESSION / composition: push each top-5 doc AWAY from the query
    #      and count how many top-5 slots open (a doc from >5 rises in).
    opened = []
    for qi in range(len(Q)):
        top5 = [int(x) for x in rank_wm[qi, :5]]
        col = s_wm[qi].copy()
        for di in top5:
            _, a_raw = targeted_adv(di, Q[qi], toward=-1.0)
            col[di] = float(unit(Q[qi]) @ unit(a_raw))
        new_top5 = set(int(x) for x in np.argsort(-col)[:5])
        opened.append(len(new_top5 - set(top5)))
    opened = np.array(opened)
    print(f"  Suppression: top-5 slots opened mean={opened.mean():.3f} "
          f"max={opened.max()}  queries w/ >=1: "
          f"{int((opened>=1).sum())}/{len(Q)} "
          f"({100*(opened>=1).mean():.2f}%)")
    results["suppression_slots_mean"] = float(opened.mean())
    results["suppression_slots_max"] = int(opened.max())
    results["suppression_queries_opened_pct"] = float(100*(opened >= 1).mean())

    # ---- Summary ---------------------------------------------------
    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  silent budget exists:  untargeted PASS {results['untargeted_pass']}"
          f", f_I inv {results['untargeted_fI_invariant']}")
    print(f"  worst-case drift:      cos>={results['untargeted_cos_min']:.4f}, "
          f"top-5 overlap {results['untargeted_top5_overlap']:.3f}, "
          f"docs unchanged {results['untargeted_docs_unchanged_pct']:.1f}%")
    print(f"  targeted injection:    random {results['injection_success']}, "
          f"best-positioned {results['injection_bestpos']} into top-5")
    print(f"  suppression:           slots opened/query mean "
          f"{results['suppression_slots_mean']:.3f}, "
          f"{results['suppression_queries_opened_pct']:.2f}% queries affected")
    print("=" * 62)
    with open("results/sub_resolution.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("Saved results/sub_resolution.json")


if __name__ == "__main__":
    main()
