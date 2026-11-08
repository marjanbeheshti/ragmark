#!/usr/bin/env python3
"""
Deletion / Censorship Attack + Signed Manifest (Walkthrough Sec. 6.7)
=====================================================================
The gap NO per-vector check can see. Level-1 has WRITE access to the
vector DB, and write access includes DELETE. To censor a document the
attacker does not need the permutation or flooding machinery -- they
simply remove its entry. Every remaining vector verifies clean, the
gate has nothing to inspect (absence is invisible to per-vector
verification BY CONSTRUCTION), and the victim query is silently served
wrong-but-clean context. The paper's tamper-detection goal ("given any
vector in the database...") is per-vector; COMPLETENESS is a separate
property, and this experiment measures its absence.

The fix is a SIGNED CORPUS MANIFEST (ragmark_hardened.build_manifest):
an HMAC over (epoch, sorted doc_id set). One mechanism closes three
holes at once:
  - deletion/censorship : periodic audit names exactly the missing ids
  - revoked-doc replay  : replayed tuple's doc_id absent from the
                          current epoch's manifest -> gate rejects
  - rollback            : stale epoch detected

  (A) Censorship-by-deletion: delete each query's GT document.
      Measure: verifier alarms (expect 0 -- undetected), recall
      collapse, fraction of queries served fully-"clean" context.
  (B) Manifest audit: detects the deletions, identifies the exact
      doc_ids (expect 100%).
  (C) Revoked replay: revoke 50 docs (new epoch), attacker re-inserts
      their genuine (vector, meta) tuples. Gate membership check
      rejects all; non-revoked docs unaffected.

Writes results/deletion_attack.json
Usage: python experiments/deletion_attack.py
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
from sklearn.decomposition import PCA

N_DOCS = 500
N_QUERIES = 500
N_REVOKED = 50
KEY = R.MASTER_KEY
SEED = 2026


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"date": datetime.now().isoformat(),
           "attack": "deletion/censorship + manifest", "seed": SEED}

    print("=" * 62)
    print("Deletion / Censorship Attack + Signed Manifest (Sec. 6.7)")
    print("=" * 62)

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

    print("  Watermarking...")
    wm, metas, doc_ids = [], [], []
    for i in range(len(E)):
        w_raw, meta = RH.watermark_hardened(E[i], i % 8, U, mu,
                                            e1_band, e0_band, e0_ppb,
                                            doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta); doc_ids.append(i)
    wm = np.asarray(wm)

    # signed manifest for epoch 1 (the full legitimate corpus)
    man1 = RH.build_manifest(doc_ids, epoch=1, key=KEY)
    assert RH.verify_manifest(man1, KEY)
    print(f"  Manifest epoch=1 signed over {man1['count']} doc_ids")

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

    # =================================================================
    # (A) Censorship by deletion: remove each query's GT document
    # =================================================================
    print("\n[A] Censorship-by-deletion (delete each query's GT doc)")
    alarms = 0            # any verifier rejection among served results
    clean_served = 0      # queries whose whole top-5 verifies PASS
    gt_recall = 0
    for qi in range(len(Q)):
        X = int(gt[qi])
        keep = [j for j in range(len(wm)) if j != X]     # X deleted
        db = wm[keep]
        s = unit(Q[qi]) @ unit(db).T
        top = np.argsort(-s)[:5]
        verdicts = [RH.verify_hardened(db[j], U, mu, e0_band, e0_ppb,
                                       meta=metas[keep[j]], key=KEY,
                                       expected_doc_id=keep[j])["verdict"]
                    for j in top]
        ok = all(v == "PASS" for v in verdicts)
        clean_served += ok
        alarms += sum(v != "PASS" for v in verdicts)
        gt_recall += any(keep[j] == X for j in top)      # always False
    print(f"  Verifier alarms raised:        {alarms}  (expect 0 --- "
          f"absence is invisible)")
    print(f"  Queries served all-PASS top-5: {clean_served}/{len(Q)}")
    print(f"  GT Recall@5 for victim query:  {gt_recall}/{len(Q)} "
          f"(was {0.92:.2f} pre-attack)")
    res["A_alarms"] = int(alarms)
    res["A_clean_served"] = f"{clean_served}/{len(Q)}"
    res["A_gt_recall"] = f"{gt_recall}/{len(Q)}"

    # =================================================================
    # (B) Manifest audit detects the deletions exactly
    # =================================================================
    print("\n[B] Manifest audit (batch deletion of 100 random docs)")
    deleted = sorted(rng.choice(len(wm), size=100, replace=False).tolist())
    remaining_ids = [i for i in doc_ids if i not in set(deleted)]
    missing, foreign = RH.audit_index(remaining_ids, man1)
    exact = (missing == deleted and foreign == [])
    print(f"  Deleted 100 docs; audit reports {len(missing)} missing, "
          f"{len(foreign)} foreign")
    print(f"  Audit identifies the exact deleted set: "
          f"{'YES' if exact else 'NO'}")
    res["B_audit_missing"] = len(missing)
    res["B_audit_exact"] = bool(exact)

    # =================================================================
    # (C) Revoked-document replay vs manifest gate
    # =================================================================
    print(f"\n[C] Revoked replay ({N_REVOKED} docs revoked at epoch 2)")
    revoked = sorted(rng.choice(len(wm), size=N_REVOKED,
                                replace=False).tolist())
    live = [i for i in doc_ids if i not in set(revoked)]
    man2 = RH.build_manifest(live, epoch=2, key=KEY)
    # attacker re-inserts the revoked docs' GENUINE tuples
    replay_pass = sum(
        RH.verify_hardened(wm[i], U, mu, e0_band, e0_ppb, meta=metas[i],
                           key=KEY, expected_doc_id=i,
                           manifest=man2)["verdict"] == "PASS"
        for i in revoked)
    live_pass = sum(
        RH.verify_hardened(wm[i], U, mu, e0_band, e0_ppb, meta=metas[i],
                           key=KEY, expected_doc_id=i,
                           manifest=man2)["verdict"] == "PASS"
        for i in live[:200])
    print(f"  Revoked tuples PASS at gate:   {replay_pass}/{N_REVOKED}  "
          f"(expect 0)")
    print(f"  Live docs PASS (no collateral): {live_pass}/200  "
          f"(expect 200)")
    res["C_revoked_pass"] = f"{replay_pass}/{N_REVOKED}"
    res["C_live_pass"] = f"{live_pass}/200"

    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  deletion is invisible per-vector: {alarms} alarms, "
          f"recall 0.92 -> 0.00 on victim queries")
    print(f"  manifest audit: exact detection = {exact}")
    print(f"  revoked replay at gate: {replay_pass}/{N_REVOKED} pass "
          f"(closed), live unaffected {live_pass}/200")
    print("=" * 62)
    with open("results/deletion_attack.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/deletion_attack.json")


if __name__ == "__main__":
    main()
