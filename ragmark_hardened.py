#!/usr/bin/env python3
"""
ragmark_hardened.py — the h_v + strict-mode + dedup gate
========================================================
The single fix that closes the silent-modification family:

  h_v        : SHA-256 over the EXACT float32 bytes of the stored
               watermarked vector, added to the HMAC-signed tuple.
               Catches sub-resolution nudges AND free-position
               permutations (any byte change -> mismatch).
  strict mode: a vector WITHOUT metadata is rejected outright.
               Catches nudge+delete, permute+delete, and rebinding
               (in a deployment where every legitimate vector was
               ingested WITH metadata, absence is itself evidence).
  dedup      : doc_id deduplication of the retrieved top-k.
               Catches duplicate flooding (which h_v/strict cannot:
               every replica is byte-genuine).
  manifest   : an HMAC-signed inventory of the current epoch's doc_id
               set. Closes the three holes no PER-VECTOR check can see:
               deletion/censorship (periodic audit names exactly the
               missing doc_ids), revoked-document replay (a replayed
               tuple's doc_id is absent from the current manifest ->
               gate rejects), and index rollback (stale epoch).

Wraps ragmark.py without modifying it. Import as:
    import ragmark_hardened as RH
    w, meta = RH.watermark_hardened(...)      # meta gains h_v + tag2
    v = RH.verify_hardened(w, U, mu, e0_band, e0_ppb, meta=meta)
    kept = RH.dedup_topk(ranked_indices, doc_ids, k)
"""
import hashlib
import hmac

import numpy as np

import ragmark as R


# ---------------------------------------------------------------------
# Exact-vector hash
# ---------------------------------------------------------------------
def hash_vector(w_raw):
    """SHA-256 over the exact float32 byte representation of the stored
    vector. Both signer and verifier cast to float32 first, so the hash
    is well-defined regardless of intermediate float64 arithmetic."""
    b = np.ascontiguousarray(np.asarray(w_raw, dtype=np.float32)).tobytes()
    return hashlib.sha256(b).hexdigest()[:32]


# ---------------------------------------------------------------------
# Extended HMAC (binds h_v into the signed tuple)
# ---------------------------------------------------------------------
def sign_meta_h(f_a, f_i, h_v, doc_id, sid, selection, key=R.MASTER_KEY):
    msg = (f"{f_a}|{f_i}|{h_v}|{doc_id}|{sid}|"
           f"{R._sel_str(selection)}").encode()
    return hmac.new(R._hmac_key(key), msg, hashlib.sha256).hexdigest()


def verify_meta_h(meta, key=R.MASTER_KEY):
    exp = sign_meta_h(meta["f_a"], meta["f_i"], meta["h_v"],
                      meta["doc_id"], meta["sid"], meta["sel"], key)
    return hmac.compare_digest(exp, meta["tag2"])


# ---------------------------------------------------------------------
# Hardened pipeline
# ---------------------------------------------------------------------
def watermark_hardened(v_raw, source_id, U, mu, e1_band, e0_band, e0_ppb,
                       doc_id=None, key=R.MASTER_KEY):
    """Same embedding as ragmark.watermark; metadata additionally carries
    h_v and an extended tag2 binding it."""
    w_raw, meta = R.watermark(v_raw, source_id, U, mu,
                              e1_band, e0_band, e0_ppb,
                              doc_id=doc_id, key=key)
    meta["h_v"] = hash_vector(w_raw)
    meta["tag2"] = sign_meta_h(meta["f_a"], meta["f_i"], meta["h_v"],
                               meta["doc_id"], meta["sid"], meta["sel"], key)
    return w_raw, meta


# ---------------------------------------------------------------------
# Signed corpus manifest (completeness / revocation / rollback)
# ---------------------------------------------------------------------
def build_manifest(doc_ids, epoch, key=R.MASTER_KEY):
    """HMAC-signed inventory of the doc_ids legitimately in the index
    at `epoch`. Revoking a document = issuing a new epoch without it."""
    ids = sorted(int(i) for i in doc_ids)
    ids_hash = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    msg = f"manifest|{epoch}|{len(ids)}|{ids_hash}".encode()
    tag = hmac.new(R._hmac_key(key), msg, hashlib.sha256).hexdigest()
    return {"epoch": int(epoch), "count": len(ids), "ids_hash": ids_hash,
            "tag": tag, "ids": set(ids)}


def verify_manifest(man, key=R.MASTER_KEY):
    ids = sorted(man["ids"])
    ids_hash = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    msg = f"manifest|{man['epoch']}|{len(ids)}|{ids_hash}".encode()
    exp = hmac.new(R._hmac_key(key), msg, hashlib.sha256).hexdigest()
    return (ids_hash == man["ids_hash"] and len(ids) == man["count"]
            and hmac.compare_digest(exp, man["tag"]))


def audit_index(present_doc_ids, man):
    """Compare the index's current doc_id set against the manifest.
    Returns (missing, foreign): manifest docs absent from the index
    (deleted/censored) and index docs absent from the manifest
    (revoked replays / unauthorized additions)."""
    present = {int(i) for i in present_doc_ids}
    return sorted(man["ids"] - present), sorted(present - man["ids"])


def verify_hardened(w_raw, U, mu, e0_band, e0_ppb, meta=None,
                    key=R.MASTER_KEY, tau=R.TAU, strict=True,
                    expected_doc_id=None, manifest=None):
    """Hardened dual-layer verification.
      1. STRICT MODE: no metadata -> REJECT (no unattributed pass).
      2. Extended HMAC (covers h_v) checked before trusting anything.
      3. h_v vs hash of the candidate's exact bytes -> catches every
         byte-level modification, including sub-resolution nudges and
         free-position permutations.
      4. Optional doc_id/index binding check: the doc_id the INDEX
         claims must equal the SIGNED doc_id (anti-rebinding when the
         attacker leaves metadata attached but rebinds the index row).
      5. Optional manifest membership: the signed doc_id must be in the
         current epoch's manifest (rejects revoked replays / rollback).
      6. Base dual-layer checks (E_I gate, E_A decode, s_A == s_I).
    """
    if meta is None:
        if strict:
            return {"verdict": "REJECTED_NO_METADATA", "source_id": None,
                    "c_i": None, "c_a": None}
        return R.verify(w_raw, U, mu, e0_band, e0_ppb, meta=None,
                        key=key, tau=tau)
    if "h_v" not in meta or "tag2" not in meta:
        return {"verdict": "REJECTED_LEGACY_META", "source_id": None,
                "c_i": None, "c_a": None}
    if not verify_meta_h(meta, key):
        return {"verdict": "TAMPERED_BAD_SIG", "source_id": None,
                "c_i": None, "c_a": None}
    if hash_vector(w_raw) != meta["h_v"]:
        return {"verdict": "REJECTED_HASH_MISMATCH", "source_id": None,
                "c_i": None, "c_a": None}
    if expected_doc_id is not None and expected_doc_id != meta["doc_id"]:
        return {"verdict": "REJECTED_DOCID_MISMATCH", "source_id": None,
                "c_i": None, "c_a": None}
    if manifest is not None and int(meta["doc_id"]) not in manifest["ids"]:
        return {"verdict": "REJECTED_NOT_IN_MANIFEST", "source_id": None,
                "c_i": None, "c_a": None}
    return R.verify(w_raw, U, mu, e0_band, e0_ppb, meta=meta,
                    key=key, tau=tau)


# ---------------------------------------------------------------------
# Metadata repair (re-signing) -- strict mode's answer to metadata loss
# ---------------------------------------------------------------------
def repair_metadata(w_raw, doc_id, U, mu, e1_band, e0_band, e0_ppb,
                    key=R.MASTER_KEY, tau=R.TAU):
    """Re-create the signed metadata for a watermarked vector that lost
    its blob. Everything is recoverable by the KEY HOLDER: the source
    ID decodes from the vector itself (E_I is self-derived), f_A
    recomputes from the raw vector, and a fresh greedy selection over
    the E_A band yields a valid sel WITHOUT touching the vector (the
    vector already encodes s, so agreeing pairs are plentiful; if the
    greedy would need a nudge -- i.e. would modify the vector -- the
    repair refuses and the document goes to re-ingestion, because any
    modification would break the E_I encoding).

    Returns (meta, status): status is "REPAIRED", or a refusal:
      "REJECTED_EI"    -- vector fails the integrity gate (never bless
                          a tampered/foreign vector)
      "NEEDS_REINGEST" -- selection would require modifying the vector
      "SELECTION_LOW"  -- fresh sel does not decode s confidently

    Honest caveat (state it wherever repair is offered): repair
    certifies the vector FROM REPAIR TIME FORWARD. Modifications inside
    E_I's blind spots (sub-resolution nudges, free-position
    permutations) made while the vector was unsigned pass the E_I gate
    and would be blessed. Repair is maintenance, not retroactive
    forensics; the defense against that window is keeping it short
    (quarantine on loss, repair from backup-verified state).
    """
    v_pc = (np.asarray(w_raw, dtype=mu.dtype) - mu) @ U.T
    s_i, c_i, verdict = R.decode_e0(v_pc, key, e0_band, e0_ppb, tau=tau)
    if verdict == "REJECTED":
        return None, "REJECTED_EI"
    f_a = R.content_fingerprint(w_raw)
    w2, sel = R.embed_e1(v_pc.copy(), s_i, f_a, key, e1_band)
    if not np.array_equal(w2, v_pc):
        return None, "NEEDS_REINGEST"
    s_a, c_a = R.decode_e1(v_pc, sel)
    if c_a < tau or s_a != s_i:
        return None, "SELECTION_LOW"
    f_i = R.full_fingerprint(v_pc)
    tag = R.sign_meta(f_a, f_i, doc_id, s_i, sel, key)
    meta = {"f_a": f_a, "f_i": f_i, "doc_id": doc_id,
            "sid": s_i, "sel": sel, "tag": tag}
    meta["h_v"] = hash_vector(w_raw)
    meta["tag2"] = sign_meta_h(f_a, f_i, meta["h_v"], doc_id, s_i, sel, key)
    return meta, "REPAIRED"


# ---------------------------------------------------------------------
# Gate-side deduplication (closes duplicate flooding)
# ---------------------------------------------------------------------
def dedup_topk(ranked_entry_indices, entry_doc_ids, k):
    """Walk a ranked list of DB-entry indices; keep the first entry per
    doc_id until k distinct documents are collected. Returns the kept
    entry indices (the deduplicated top-k)."""
    seen, kept = set(), []
    for idx in ranked_entry_indices:
        did = entry_doc_ids[idx]
        if did in seen:
            continue
        seen.add(did)
        kept.append(idx)
        if len(kept) == k:
            break
    return kept
