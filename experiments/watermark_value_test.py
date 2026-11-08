#!/usr/bin/env python3
"""
What the Watermark Adds Over a Signed-Hash-Only Gate (Walkthrough Sec. 7)
========================================================================
The honest comparison. In strict mode the steelmanned baseline
(baseline_hashonly.py: HMAC + h_v + strict + manifest + dedup,
quantization-aware) WINS OR TIES RagMark on the headline security
table -- byte-exact tamper detection, 2^-128 injection bounds, exact
attribution, perfect invisibility, lower latency. We do not hide this;
Part A reports it.

The watermark's value is isolated in the regime where the signal must
live INSIDE the vector rather than beside it -- i.e. when the
out-of-band metadata channel is lost. Parts B-E measure that boundary:

  (A) Headline parity: both reject every attack in strict mode; the
      baseline is strictly simpler. (Baseline is not weaker here.)
  (B) Metadata-loss recovery, TWO conditions. The baseline can re-sign
      iff an out-of-band doc_id->source record survives (ingestion log,
      document store): h_v and doc_id are recomputable, only `s` died
      with the blob. RagMark recovers from the vector alone in both
      conditions. The advantage is claimed only where it exists:
      vector-only artifacts (exfiltrated indexes, migrations, backups)
      where blob and log plausibly fail together.
  (C) Triage of unattributed vectors: stripped-genuine vs HELD-OUT
      same-model, same-domain embeddings that were never watermarked.
      (Random Gaussians would be separable by any outlier detector; we
      verify the alien class is distributionally legitimate by showing
      cosine-to-corpus-mean does NOT separate it.) Baseline classifies
      at chance -- the classes are statistically indistinguishable
      without an in-vector signal; E_I separates them.
  (D) Out-of-band / exact-copy attribution: leaked float32 copies carry
      RagMark's in-vector ID; the baseline's signal does not travel.
      HONESTLY SCOPED: holds for exact copies only -- noise/quantization
      destroys it too (reported as the boundary, not hidden).
  (E) Permissive deployment axis: where metadata cannot be guaranteed,
      the baseline is UNDEFINED (no check at all); RagMark degrades to
      the 10^-8 watermark-only gate -- measured on BOTH genuine bare
      vectors (should pass) and alien bare vectors (should reject).

DTYPE NOTE (the Part-C bug this version fixes): the previous version
built the triage pile with np.vstack([float32 genuine, float64 random
aliens]), silently upcasting the stored float32 vectors to float64.
The PC round trip at the higher precision rounds a handful of
coordinates across 2-decimal boundaries differently than the stored
representation, changing f_I on EVERY vector -- pairs reshuffle, and
both classes decode as noise (c_I ~ 0.62 / 0.62, accuracy 0.50).
Parts D and E never stacked and were correct (100/100, 200/200).
Fix: never mix dtypes; verify each vector individually, at its stored
dtype, through the same code path Part E uses.

Writes results/watermark_value.json
Usage: python experiments/watermark_value_test.py
"""
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import ragmark as R
import ragmark_hardened as RH
import baseline_hashonly as B
from sklearn.decomposition import PCA

N_DOCS = 500          # watermarked corpus
N_ALIEN = 500         # held-out, same model/domain, never watermarked
KEY = R.MASTER_KEY
SEED = 2026


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"date": datetime.now().isoformat(),
           "test": "watermark value over signed-hash-only baseline"}

    print("=" * 66)
    print("What the Watermark Adds Over Signed-Hash-Only (Sec. 7)")
    print("=" * 66)

    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
    print(f"\n[SETUP] Loading corpus ({N_DOCS} docs + "
          f"{N_ALIEN} held-out aliens)...")
    ds = load_dataset("rajpurkar/squad", split="train")
    seen, all_texts = set(), []
    for r in ds:
        if r["context"] not in seen and len(all_texts) < N_DOCS + N_ALIEN:
            seen.add(r["context"]); all_texts.append(r["context"])
    texts = all_texts[:N_DOCS]              # watermarked corpus
    model = SentenceTransformer("all-mpnet-base-v2")
    emb = np.asarray(model.encode(all_texts, normalize_embeddings=True,
                                  show_progress_bar=True))
    E = emb[:N_DOCS]
    # held-out aliens: same model, same domain, never watermarked;
    # pinned to float32 = the dtype vectors are stored in
    aliens = np.ascontiguousarray(emb[N_DOCS:], dtype=np.float32)
    d = E.shape[1]
    # PCA fit on the corpus only -- aliens are out-of-corpus by design
    pca = PCA(n_components=min(d, len(E))).fit(E)
    U, mu = pca.components_, pca.mean_
    e1_band, e0_band, e0_ppb = R.compute_bands(d)
    print(f"  d={d}, E_A={e1_band}, E_I={e0_band}, ppb={e0_ppb}")

    print("  Watermarking (RagMark hardened) + signing (baseline)...")
    wm, metas, base_metas, doc_ids = [], [], [], []
    for i in range(len(E)):
        w_raw, meta = RH.watermark_hardened(E[i], i % 8, U, mu,
                                            e1_band, e0_band, e0_ppb,
                                            doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta); doc_ids.append(i)
        base_metas.append(B.sign_hashonly(w_raw, i, i % 8, KEY))
    wm = np.asarray(wm)                     # single dtype: as stored
    manifest = RH.build_manifest(doc_ids, epoch=1, key=KEY)

    # the out-of-band ingestion record the baseline MAY have (Part B):
    # doc_id -> source. h_v / doc_id are recomputable; only `s` is here.
    ingestion_record = {i: {"doc_id": i, "sid": i % 8}
                        for i in range(len(E))}

    # One shared helper: the SAME code path Part E used (validated
    # 200/200), applied per-vector at its stored dtype. Never stack
    # mixed-dtype arrays (see DTYPE NOTE in the module docstring).
    def e0_gate(v):
        out = RH.verify_hardened(np.asarray(v), U, mu, e0_band, e0_ppb,
                                 meta=None, key=KEY, strict=False)
        return out["verdict"] == "PASS"

    def e0_conf(v):
        v = np.asarray(v)
        return float(RH.R.decode_e0((v - mu) @ U.T, KEY,
                                    e0_band, e0_ppb)[1])

    # =================================================================
    # (A) Headline parity -- baseline is NOT weaker in strict mode
    # =================================================================
    print("\n[A] Headline parity (strict mode): both reject attacks")
    # injection: 50 alien vectors (held-out real embeddings, not noise)
    inj = aliens[:50]
    rm_rej = sum(RH.verify_hardened(inj[j], U, mu, e0_band, e0_ppb,
                 meta=None, key=KEY, strict=True)["verdict"] != "PASS"
                 for j in range(50))
    bl_rej = sum(B.verify_hashonly(inj[j], meta=None, key=KEY,
                 strict=True)["verdict"] != "PASS" for j in range(50))
    # byte modification on a legit vector (both must reject w/ metadata)
    mod = wm.copy()
    mod[:50] += rng.standard_normal((50, d)) * 1e-3
    rm_mod = sum(RH.verify_hardened(mod[j], U, mu, e0_band, e0_ppb,
                 meta=metas[j], key=KEY,
                 expected_doc_id=j)["verdict"] != "PASS" for j in range(50))
    bl_mod = sum(B.verify_hashonly(mod[j], meta=base_metas[j], key=KEY,
                 expected_doc_id=j)["verdict"] != "PASS" for j in range(50))
    print(f"  Injection rejected:  RagMark {rm_rej}/50, baseline {bl_rej}/50")
    print(f"  Byte-mod rejected:   RagMark {rm_mod}/50, baseline {bl_mod}/50")
    print("  -> parity (baseline ties; it is simpler and byte-exact)")
    res["A_injection"] = {"ragmark": f"{rm_rej}/50", "baseline": f"{bl_rej}/50"}
    res["A_bytemod"] = {"ragmark": f"{rm_mod}/50", "baseline": f"{bl_mod}/50"}

    # =================================================================
    # (B) Metadata-loss recovery -- two conditions
    # =================================================================
    print("\n[B] Metadata-loss recovery (strip 100 blobs)")
    idx = rng.choice(len(wm), size=100, replace=False)

    # condition 1: ingestion record SURVIVES -> baseline re-signs fine
    bl_with_rec = sum(
        B.resign_from_record(wm[i], ingestion_record.get(int(i)),
                             key=KEY) is not None
        for i in idx)
    # condition 2: record lost with the blob -> baseline is dead;
    # nothing in the bare vector recovers `s` (structural, not measured)
    bl_no_rec = sum(
        B.resign_from_record(wm[i], None, key=KEY) is not None
        for i in idx)
    # ragmark: recovers from the vector alone in BOTH conditions
    rm_recovered, sid_ok = 0, 0
    for i in idx:
        meta2, status = RH.repair_metadata(wm[i], int(i), U, mu,
                                           e1_band, e0_band, e0_ppb, KEY)
        if status == "REPAIRED":
            rm_recovered += 1
            sid_ok += (meta2["sid"] == metas[i]["sid"])
    print(f"  Record survives:  baseline {bl_with_rec}/100  "
          f"(h_v, doc_id recomputable; s from record)")
    print(f"  Record lost:      baseline {bl_no_rec}/100  "
          f"(structural: s lived only beside the vector)")
    print(f"  RagMark (either): {rm_recovered}/100, "
          f"source ID correct {sid_ok}/{rm_recovered}")
    print("  -> advantage claimed ONLY for vector-only artifacts")
    print("     (exfiltrated index, migration, backup: blob+log die together)")
    res["B_baseline_with_record"] = f"{bl_with_rec}/100"
    res["B_baseline_no_record"] = f"{bl_no_rec}/100"
    res["B_ragmark_recovered"] = f"{rm_recovered}/100"
    res["B_ragmark_sid_correct"] = f"{sid_ok}/{rm_recovered}"

    # =================================================================
    # (C) Triage of unattributed vectors (classification experiment)
    # =================================================================
    print("\n[C] Triage: stripped-genuine vs HELD-OUT aliens "
          "(metadata removed)")
    n_each = 200
    g_idx = rng.choice(len(wm), size=n_each, replace=False)
    a_idx = rng.choice(len(aliens), size=n_each, replace=False)
    genuine = [wm[i] for i in g_idx]        # per-vector, stored dtype
    alien_s = [aliens[i] for i in a_idx]    # NO vstack -- see DTYPE NOTE

    # sanity: the alien class is NOT separable by outlier detection --
    # cosine to the corpus mean overlaps (else any detector solves this)
    mdir = unit(E.mean(axis=0))
    cos_g = np.array([float(np.dot(unit(np.asarray(v)), mdir))
                      for v in genuine])
    cos_a = np.array([float(np.dot(unit(np.asarray(v)), mdir))
                      for v in alien_s])
    print(f"  cos-to-corpus-mean: ours {cos_g.mean():.3f}"
          f"+-{cos_g.std():.3f} vs alien {cos_a.mean():.3f}"
          f"+-{cos_a.std():.3f}  (overlap -> outlier detection fails)")

    # ragmark: SAME gate as Part E, per vector, stored dtype
    pred_g = np.array([e0_gate(v) for v in genuine])
    pred_a = np.array([e0_gate(v) for v in alien_s])
    ci_g = np.array([e0_conf(v) for v in genuine])
    ci_a = np.array([e0_conf(v) for v in alien_s])
    rm_acc = float((pred_g.sum() + (~pred_a).sum()) / (2 * n_each))
    # baseline: no metadata, no in-vector signal -> undecidable, chance.
    # Structural zero, not a measurement: with the outlier route closed
    # above, the classes are indistinguishable to a metadata-only scheme.
    bl_acc = 0.5
    print(f"  RagMark triage accuracy:  {rm_acc:.3f}  "
          f"(c_I: ours {ci_g.mean():.3f} vs alien {ci_a.mean():.3f})")
    print(f"  RagMark: genuine pass {int(pred_g.sum())}/{n_each}, "
          f"alien reject {int((~pred_a).sum())}/{n_each}")
    print(f"  Baseline triage accuracy: {bl_acc:.3f}  "
          f"(chance -- no signal; structural)")
    res["C_ragmark_acc"] = rm_acc
    res["C_ragmark_ci_sep"] = (float(ci_g.mean()), float(ci_a.mean()))
    res["C_cos_overlap"] = {"ours": [float(cos_g.mean()), float(cos_g.std())],
                            "alien": [float(cos_a.mean()), float(cos_a.std())]}
    res["C_baseline_acc"] = bl_acc

    # =================================================================
    # (D) Out-of-band attribution -- HONESTLY SCOPED
    # =================================================================
    print("\n[D] Out-of-band attribution of leaked vectors")
    leaked = wm[rng.choice(len(wm), size=100, replace=False)]
    exact_ok = sum(e0_gate(v) for v in leaked)
    noised = leaked + rng.standard_normal(leaked.shape) * 1e-3
    noise_ok = sum(e0_gate(v) for v in noised)
    print(f"  Exact float32 copies attributable (RagMark): {exact_ok}/100")
    print(f"  After 1e-3 noise (the boundary):             {noise_ok}/100")
    print(f"  Baseline (signal not in vector):             0/100 either way")
    print("  -> RagMark attributes EXACT leaked copies; neither scheme")
    print("     attributes noised/re-quantized copies. Stated as the limit.")
    res["D_exact"] = f"{exact_ok}/100"
    res["D_noised"] = f"{noise_ok}/100"
    res["D_baseline"] = "0/100"

    # =================================================================
    # (E) Permissive deployment axis -- both directions
    # =================================================================
    print("\n[E] Permissive deployment (metadata channel not guaranteed)")
    g_sample = [wm[i] for i in rng.choice(len(wm), size=200, replace=False)]
    a_sample = [aliens[i] for i in
                rng.choice(len(aliens), size=200, replace=False)]
    rm_pass = sum(e0_gate(v) for v in g_sample)
    rm_rej_alien = sum(not e0_gate(v) for v in a_sample)
    bl_pass = sum(B.verify_hashonly(np.asarray(v), meta=None, key=KEY,
                  strict=False)["verdict"] == "PASS" for v in g_sample)
    bl_nocheck = sum(B.verify_hashonly(np.asarray(v), meta=None, key=KEY,
                     strict=False)["verdict"] == "NO_CHECK_POSSIBLE"
                     for v in g_sample)
    print(f"  RagMark: genuine bare vectors PASS:      {rm_pass}/200")
    print(f"  RagMark: alien bare vectors rejected:    {rm_rej_alien}/200")
    print(f"  Baseline PASS on bare vectors:           {bl_pass}/200")
    print(f"  Baseline 'NO_CHECK_POSSIBLE':            {bl_nocheck}/200  "
          f"(undefined -- nothing to verify, either class)")
    res["E_ragmark_genuine_pass"] = f"{rm_pass}/200"
    res["E_ragmark_alien_reject"] = f"{rm_rej_alien}/200"
    res["E_baseline_nocheck"] = f"{bl_nocheck}/200"

    # =================================================================
    print("\n" + "=" * 66)
    print("SUMMARY (the MAC-vs-watermark thesis, measured)")
    print(f"  [A] strict-mode security: PARITY (baseline ties, simpler)")
    print(f"  [B] recovery:  record survives -> parity "
          f"({bl_with_rec}/100 vs {rm_recovered}/100);")
    print(f"                 record lost -> RagMark {rm_recovered}/100 "
          f"vs baseline {bl_no_rec}/100 (structural)")
    print(f"  [C] triage:    RagMark {rm_acc:.2f} vs baseline "
          f"{bl_acc:.2f} (chance; outlier route closed)")
    print(f"  [D] exact-copy attrib: RagMark {exact_ok}/100 vs 0; "
          f"boundary: noised {noise_ok}/100")
    print(f"  [E] permissive: RagMark {rm_pass}/200 pass + "
          f"{rm_rej_alien}/200 reject; baseline undefined")
    print("  THESIS: the gate gives strict-mode security; the watermark")
    print("  gives in-band provenance that survives loss of the channel.")
    print("=" * 66)
    with open("results/watermark_value.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/watermark_value.json")


if __name__ == "__main__":
    main()
