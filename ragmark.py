"""
ragmark.py — RAGMARK single source of truth
============================================
Hybrid embedding watermark for RAG vector databases.

  Attribution Layer (E_A): greedy pool-based selection in band [l1,h1).
      Pairs are chosen by natural gap and CANNOT be re-derived from the
      key alone, so the selection is stored and HMAC-signed.
  Integrity Layer (E_I): swap-mode in band [l2,h2) with a self-derived
      FULL-VECTOR fingerprint (hashes all d PC coordinates).

Metadata is authenticated with HMAC-SHA-256 under an independent
signing key K' = SHA-256(K || "|hmac"), binding
(f_A, f_I, doc_id, source_id, sel). Verification checks the tag
(constant-time) BEFORE trusting any stored selection.

This module is the ONE implementation; all experiment scripts import
from it. Numbers in the paper are produced by running the scripts in
experiments/ against this core.
"""
import hashlib
import hmac

import numpy as np

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
MASTER_KEY = "ragmark_master_secret_2026"
N_BITS = 8
PPB_E1 = 3          # pairs per bit, Attribution (greedy)
DELTA = 0.02        # min gap forced when a selected E_A pair disagrees
TAU = 0.85          # verification threshold
FP_TOP = 200        # top-magnitude raw dims for the stored fingerprint

# Domain-separation tags (independent PRNG streams per layer)
TAG_ATTR = "attr"
TAG_INTEG = "integ"


# ---------------------------------------------------------------------
# Bands (fractional rule, round())
# ---------------------------------------------------------------------
def compute_bands(d):
    """E_A = [round(.04d), round(.2d)); E_I = [round(.2d), round(.5d)).
    P2 capped at 10, reduced for a narrow band."""
    e1_lo = max(1, int(round(0.04 * d)))
    e1_hi = int(round(0.2 * d))
    e0_lo = e1_hi
    e0_hi = int(round(0.5 * d))
    e0_ppb = min(10, (e0_hi - e0_lo) // (2 * N_BITS))
    return (e1_lo, e1_hi), (e0_lo, e0_hi), e0_ppb


# ---------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------
def content_fingerprint(v_raw):
    """f_A: sign pattern of the top-FP_TOP magnitude RAW dims.
    Public (no key). Sole role: content-binding. 64-bit."""
    top = np.argsort(-np.abs(v_raw))[:FP_TOP]
    signs = tuple(1 if v_raw[d] > 0 else 0 for d in top)
    return hashlib.sha256(str(signs).encode()).hexdigest()[:16]


def full_fingerprint(v_pc):
    """f_I: sorted multiset of ALL d PC coords, rounded to 2 dp,
    signed zeros canonicalized. Swap-invariant. 64-bit."""
    c = np.round(v_pc, 2) + 0.0
    return hashlib.sha256(str(tuple(np.sort(c))).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------
# Keyed pair derivation (domain-separated, SHA-256 -> PCG64)
# ---------------------------------------------------------------------
def derive_pairs(key, tag, fingerprint, band, ppb, n_bits=N_BITS):
    """Partition band indices into ppb disjoint pairs per bit.
    Seed = SHA-256(key || tag || fingerprint), 64-bit -> PCG64."""
    seed = int.from_bytes(
        hashlib.sha256(f"{key}|{tag}|{fingerprint}".encode()).digest()[:8],
        "big")
    rng = np.random.default_rng(seed)
    idx = list(range(band[0], band[1]))
    rng.shuffle(idx)
    need = n_bits * ppb * 2
    sel = idx[:need]
    pairs = []
    for b in range(n_bits):
        bp = [(sel[(b * ppb + p) * 2], sel[(b * ppb + p) * 2 + 1])
              for p in range(ppb)]
        pairs.append(bp)
    return pairs


# ---------------------------------------------------------------------
# Attribution Layer (E_A): greedy pool-based selection
# ---------------------------------------------------------------------
def embed_e1(v_pc, source_id, f_a, key, band,
             ppb=PPB_E1, n_bits=N_BITS, delta=DELTA):
    """Greedy: each bit draws pairs from the full pool of uncommitted
    band indices, picks the ppb with the largest natural gap in the
    bit's direction, forces only disagreeing pairs (midpoint +/- delta).
    Returns (w, selection) where selection[b] = list of (a,b) pairs.
    The selection is NOT re-derivable and must be stored + signed."""
    w = v_pc.copy()
    seed = int(hashlib.sha256(
        (key + "|" + TAG_ATTR + "|" + f_a).encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    all_idx = list(range(band[0], band[1]))
    rng.shuffle(all_idx)
    committed = set()
    selection = []
    for b in range(n_bits):
        bit = (source_id >> b) & 1
        avail = [i for i in all_idx if i not in committed]
        cands = []
        for ci in range(len(avail) // 2):
            a, b_ = avail[2 * ci], avail[2 * ci + 1]
            gap = (w[a] - w[b_]) if bit == 1 else (w[b_] - w[a])
            cands.append((gap, a, b_))
        cands.sort(key=lambda x: -x[0])
        chosen = []
        for gap, a, b_ in cands[:ppb]:
            committed.add(a); committed.add(b_)
            chosen.append((a, b_))
            if gap < 0:
                mid = (w[a] + w[b_]) / 2
                if bit == 1:
                    w[a], w[b_] = mid + delta, mid - delta
                else:
                    w[a], w[b_] = mid - delta, mid + delta
        selection.append(chosen)
    return w, selection


def decode_e1(v_pc, selection, ppb=PPB_E1, n_bits=N_BITS):
    """Decode E_A from the stored selection (majority vote per bit)."""
    bits, confs = [], []
    for b in range(n_bits):
        votes = [1 if v_pc[a] > v_pc[b_] else 0
                 for a, b_ in selection[b] if v_pc[a] != v_pc[b_]]
        if not votes:
            bits.append(0); confs.append(0.0)
        else:
            ones = sum(votes)
            bits.append(1 if ones > len(votes) / 2 else 0)
            confs.append(max(ones, len(votes) - ones) / ppb)
    sid = sum(bit << j for j, bit in enumerate(bits))
    return sid, float(np.mean(confs))


# ---------------------------------------------------------------------
# Integrity Layer (E_I): swap-mode, full-vector fingerprint
# ---------------------------------------------------------------------
def embed_e0(v_pc, source_id, key, band, ppb, n_bits=N_BITS):
    """Swap-mode encode in [l2,h2). Pairs derived from the full-vector
    fingerprint of the (post-E_A) vector. Preserves the multiset."""
    w = v_pc.copy()
    f_i = full_fingerprint(v_pc)
    pairs = derive_pairs(key, TAG_INTEG, f_i, band, ppb, n_bits)
    for b in range(n_bits):
        bit = (source_id >> b) & 1
        for a, b_ in pairs[b]:
            if bit == 1 and w[a] < w[b_]:
                w[a], w[b_] = w[b_], w[a]
            elif bit == 0 and w[a] > w[b_]:
                w[a], w[b_] = w[b_], w[a]
    return w


def decode_e0(v_pc, key, band, ppb, n_bits=N_BITS, tau=TAU):
    """Decode E_I. Recomputes the full-vector fingerprint from the
    candidate vector, so any coordinate change reshuffles pairs and
    collapses confidence -> rejection."""
    f_i = full_fingerprint(v_pc)
    pairs = derive_pairs(key, TAG_INTEG, f_i, band, ppb, n_bits)
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
    sid = sum(bit << j for j, bit in enumerate(bits))
    conf = float(np.mean(confs))
    return sid, conf, ("PASS" if conf >= tau else "REJECTED")


# ---------------------------------------------------------------------
# HMAC metadata (independent signing key K')
# ---------------------------------------------------------------------
def _hmac_key(key=MASTER_KEY):
    return hashlib.sha256((key + "|hmac").encode()).digest()


def _sel_str(selection):
    # canonical serialization of the greedy (a,b) pairs
    return ";".join(",".join(f"{a}-{b}" for a, b in bit) for bit in selection)


def sign_meta(f_a, f_i, doc_id, sid, selection, key=MASTER_KEY):
    msg = f"{f_a}|{f_i}|{doc_id}|{sid}|{_sel_str(selection)}".encode()
    return hmac.new(_hmac_key(key), msg, hashlib.sha256).hexdigest()


def verify_meta(f_a, f_i, doc_id, sid, selection, tag, key=MASTER_KEY):
    exp = sign_meta(f_a, f_i, doc_id, sid, selection, key)
    return hmac.compare_digest(exp, tag)


# ---------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------
def watermark(v_raw, source_id, U, mu, e1_band, e0_band, e0_ppb,
              doc_id=None, key=MASTER_KEY):
    """Embed both layers, return (w_raw, metadata).
    metadata = dict(f_a, f_i, doc_id, sid, sel, tag)."""
    if doc_id is None:
        doc_id = source_id
    v_pc = (np.asarray(v_raw, dtype=mu.dtype) - mu) @ U.T
    f_a = content_fingerprint(v_raw)
    v_pc, sel = embed_e1(v_pc, source_id, f_a, key, e1_band)
    f_i = full_fingerprint(v_pc)                       # after E_A
    v_pc = embed_e0(v_pc, source_id, key, e0_band, e0_ppb)
    w_raw = v_pc @ U + mu
    tag = sign_meta(f_a, f_i, doc_id, source_id, sel, key)
    meta = {"f_a": f_a, "f_i": f_i, "doc_id": doc_id,
            "sid": source_id, "sel": sel, "tag": tag}
    return w_raw, meta


def verify(w_raw, U, mu, e0_band, e0_ppb, meta=None, key=MASTER_KEY, tau=TAU):
    """Dual-layer verification.
      - Integrity check ALWAYS runs first (E_I confidence gate).
      - If metadata present: check HMAC tag (constant-time) before
        trusting sel, decode E_A, require s_A == s_I.
    Returns dict(verdict, source_id, c_i, c_a)."""
    v_pc = (np.asarray(w_raw, dtype=mu.dtype) - mu) @ U.T
    s_i, c_i, verdict_i = decode_e0(v_pc, key, e0_band, e0_ppb, tau=tau)
    if verdict_i == "REJECTED":
        return {"verdict": "REJECTED", "source_id": None,
                "c_i": c_i, "c_a": None}
    if meta is not None:
        if not verify_meta(meta["f_a"], meta["f_i"], meta["doc_id"],
                           meta["sid"], meta["sel"], meta["tag"], key):
            return {"verdict": "TAMPERED_BAD_SIG", "source_id": None,
                    "c_i": c_i, "c_a": None}
        s_a, c_a = decode_e1(v_pc, meta["sel"])
        if c_a < tau:
            return {"verdict": "TAMPERED", "source_id": None,
                    "c_i": c_i, "c_a": c_a}
        if s_a != s_i:
            return {"verdict": "SOURCE_MISMATCH", "source_id": None,
                    "c_i": c_i, "c_a": c_a}
        return {"verdict": "PASS", "source_id": s_a, "c_i": c_i, "c_a": c_a}
    # no metadata: E_I-only (tamper detection without attribution)
    return {"verdict": "PASS", "source_id": s_i, "c_i": c_i, "c_a": None}
