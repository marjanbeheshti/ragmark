#!/usr/bin/env python3
"""
baseline_hashonly.py -- the steelmanned signed-hash-only baseline
=================================================================
RagMark's hardened gate MINUS the watermark. This is Algorithm 2 with
the E_A / E_I layers and the PCA machinery deleted, everything else
kept: per-vector HMAC over (h_v, doc_id, s), strict mode, the signed
manifest, and doc_id dedup. Same key infrastructure, same deployment
policy, same attack pipeline.

This baseline is deliberately NOT handicapped. In strict mode it wins
or ties RagMark on the headline security table:
  - tamper detection : byte-exact (no multiset class, no sub-resolution
                       slack, no free-position permutations)
  - injection        : rejected with 2^-128 bounds (forging HMAC)
  - attribution      : exact (s is a signed field)
  - invisibility     : perfect by vacuity (zero distortion)
  - latency          : one hash, no PCA transform/decode
  - quantization     : h_v computed over the STORED representation
                       (quantization-aware), so it survives compressed
                       deployments better than E_I, not worse.

The point of the comparison is therefore NOT "RagMark catches attacks
the baseline misses." It is that the baseline is CAPABILITY-BOUNDED:
its signal lives BESIDE the vector, in metadata, so it dies the moment
the metadata channel does. watermark_value_test.py measures exactly
that boundary (recovery, triage, out-of-band attribution, permissive
deployments) -- the regime where a mark inside the vector is the only
thing that can work.
"""
import hashlib
import hmac

import numpy as np

import ragmark as R
import ragmark_hardened as RH


# ---------------------------------------------------------------------
# Quantization-aware exact-byte hash (steelman: hash the STORED repr)
# ---------------------------------------------------------------------
def hash_stored(w_raw, dtype=np.float32):
    """SHA-256 over the exact bytes of the stored representation. For a
    compressed deployment pass dtype=np.float16 at BOTH sign and verify;
    the hash then matches the stored codes, so the baseline survives
    quantization (RagMark's own future-work idea, granted to the
    baseline up front)."""
    b = np.ascontiguousarray(np.asarray(w_raw, dtype=dtype)).tobytes()
    return hashlib.sha256(b).hexdigest()[:32]


# ---------------------------------------------------------------------
# Sign / verify (no watermark, no PCA, no decode)
# ---------------------------------------------------------------------
def sign_hashonly(w_raw, doc_id, s, key=R.MASTER_KEY, dtype=np.float32):
    """Per-vector signed metadata: HMAC over (h_v, doc_id, s). s is a
    plain signed field -- attribution is exact and free."""
    h_v = hash_stored(w_raw, dtype)
    msg = f"hashonly|{h_v}|{doc_id}|{s}".encode()
    tag = hmac.new(R._hmac_key(key), msg, hashlib.sha256).hexdigest()
    return {"h_v": h_v, "doc_id": doc_id, "sid": s, "tag": tag}


def verify_hashonly(w_raw, meta=None, key=R.MASTER_KEY, strict=True,
                    expected_doc_id=None, manifest=None, dtype=np.float32):
    """Baseline verification: strict mode + HMAC + h_v + doc_id binding
    + manifest membership. No vector is ever decoded; absent metadata is
    rejected under strict mode and there is NO fallback check, because
    without the watermark there is nothing in the vector to check."""
    if meta is None:
        return {"verdict": "REJECTED_NO_METADATA" if strict
                else "NO_CHECK_POSSIBLE", "source_id": None}
    exp = hmac.new(
        R._hmac_key(key),
        f"hashonly|{meta['h_v']}|{meta['doc_id']}|{meta['sid']}".encode(),
        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(exp, meta["tag"]):
        return {"verdict": "TAMPERED_BAD_SIG", "source_id": None}
    if hash_stored(w_raw, dtype) != meta["h_v"]:
        return {"verdict": "REJECTED_HASH_MISMATCH", "source_id": None}
    if expected_doc_id is not None and expected_doc_id != meta["doc_id"]:
        return {"verdict": "REJECTED_DOCID_MISMATCH", "source_id": None}
    if manifest is not None and int(meta["doc_id"]) not in manifest["ids"]:
        return {"verdict": "REJECTED_NOT_IN_MANIFEST", "source_id": None}
    return {"verdict": "PASS", "source_id": meta["sid"]}


# ---------------------------------------------------------------------
# The capability boundary, made explicit in code
# ---------------------------------------------------------------------
def resign_from_record(w_raw, record, key=R.MASTER_KEY, dtype=np.float32):
    """Re-sign a vector whose metadata blob was lost, GIVEN an
    out-of-band doc_id->source record (ingestion log, document store).
    h_v is recomputable from the stored vector and doc_id comes from
    the index binding -- the only thing that died with the blob is `s`.
    With a record the baseline recovers fully (condition 1 of Part B);
    with record=None it cannot (condition 2): no in-vector signal
    recovers `s`. That second condition -- vector-only artifacts where
    blob and log fail together -- is the capability the watermark
    supplies and the baseline structurally lacks."""
    if record is None:
        return None
    return sign_hashonly(w_raw, record["doc_id"], record["sid"],
                         key=key, dtype=dtype)


def can_resign_without_text(w_raw):
    """Deprecated: kept for compatibility. Use resign_from_record,
    which models the baseline honestly (recovery is possible whenever
    an out-of-band doc_id->source record survives, not only via the
    text). Equivalent to resign_from_record(w_raw, None) is None."""
    return False


def triage_unattributed(w_raw):
    """The baseline cannot tell a stripped-but-genuine vector from an
    alien one: with no metadata and no in-vector signal, both are just
    numbers. Classifies at chance. Returns None (undecidable)."""
    return None
