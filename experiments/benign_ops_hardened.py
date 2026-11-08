#!/usr/bin/env python3
"""
Benign-Operations False-Reject Test for the Hardened Verifier
=============================================================
The fix's own cost, measured. h_v is byte-exact, so the deployability
question is: which BENIGN infrastructure operations (storage round
trips, serialization, index add/reconstruct) preserve the float32 bytes
and which do not? Each op is applied to legitimate watermarked vectors,
then verified with BOTH the base verifier and the hardened one.

Expected: float32-preserving paths pass both (no false rejects);
float16 storage fails h_v by design (documented as the attribution-only
regime -- verify at ingestion, E_A path afterward, per the paper's
quantization discussion).

Writes results/benign_ops_hardened.json
Usage: python experiments/benign_ops_hardened.py
"""
import io
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
KEY = R.MASTER_KEY


# --------------------------- benign operations -----------------------
def op_identity(v):
    return v.copy()


def op_f32_roundtrip(v):
    """The canonical storage path: float64 pipeline -> float32 DB ->
    float64 compute."""
    return v.astype(np.float32).astype(np.float64)


def op_npy_roundtrip(v):
    """Serialize to .npy bytes (float32) and back."""
    buf = io.BytesIO()
    np.save(buf, v.astype(np.float32))
    buf.seek(0)
    return np.load(buf).astype(np.float64)


def op_json_roundtrip(v):
    """JSON list round trip of the float32 values."""
    s = json.dumps([float(x) for x in v.astype(np.float32)])
    return np.asarray(json.loads(s), dtype=np.float32).astype(np.float64)


def op_bytes_roundtrip(v):
    """Raw-bytes store/retrieve (what a binary KV store does)."""
    b = v.astype(np.float32).tobytes()
    return np.frombuffer(b, dtype=np.float32).astype(np.float64)


def op_f16_storage(v):
    """float16 storage (2x compression). Expected h_v mismatch."""
    return v.astype(np.float16).astype(np.float64)


def op_faiss_roundtrip_factory():
    """FAISS IndexFlatIP add + reconstruct, if faiss is installed."""
    try:
        import faiss  # noqa
    except Exception:
        return None

    def op(v):
        import faiss
        idx = faiss.IndexFlatIP(v.shape[0])
        idx.add(v.astype(np.float32)[None, :])
        return idx.reconstruct(0).astype(np.float64)
    return op


def main():
    os.makedirs("results", exist_ok=True)
    res = {"date": datetime.now().isoformat(),
           "test": "benign-ops false rejects, hardened verifier"}

    print("=" * 62)
    print("Benign-Operations False-Reject Test (hardened verifier)")
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

    print("  Watermarking (hardened)...")
    wm, metas = [], []
    for i in range(len(E)):
        w_raw, meta = RH.watermark_hardened(E[i], i % 8, U, mu,
                                            e1_band, e0_band, e0_ppb,
                                            doc_id=i, key=KEY)
        wm.append(w_raw); metas.append(meta)
    wm = np.asarray(wm)

    ops = [("identity (in-memory)", op_identity),
           ("float32 round trip", op_f32_roundtrip),
           (".npy save/load (f32)", op_npy_roundtrip),
           ("JSON round trip (f32)", op_json_roundtrip),
           ("raw bytes store (f32)", op_bytes_roundtrip),
           ("float16 storage", op_f16_storage)]
    faiss_op = op_faiss_roundtrip_factory()
    if faiss_op is not None:
        ops.insert(5, ("FAISS add+reconstruct", faiss_op))
    else:
        print("  (faiss not installed -- skipping FAISS row)")

    print(f"\n{'operation':26s} {'base PASS':>12s} {'hardened PASS':>14s}")
    print("-" * 56)
    for name, op in ops:
        base = hard = 0
        for i in range(len(wm)):
            v = op(wm[i])
            b = R.verify(v, U, mu, e0_band, e0_ppb,
                         meta=metas[i], key=KEY)["verdict"]
            h = RH.verify_hardened(v, U, mu, e0_band, e0_ppb,
                                   meta=metas[i], key=KEY,
                                   expected_doc_id=metas[i]["doc_id"]
                                   )["verdict"]
            base += (b == "PASS")
            hard += (h == "PASS")
        print(f"{name:26s} {base:>7d}/{len(wm)} {hard:>9d}/{len(wm)}")
        key_ = name.replace(" ", "_")
        res[f"{key_}_base"] = f"{base}/{len(wm)}"
        res[f"{key_}_hardened"] = f"{hard}/{len(wm)}"

    print("\nReading: every float32-preserving row should show hardened")
    print("PASS equal to base PASS (the fix adds NO false rejects on the")
    print("supported storage path). float16 fails h_v by design: that is")
    print("the attribution-only regime, stated in the paper, not a bug.")
    with open("results/benign_ops_hardened.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/benign_ops_hardened.json")


if __name__ == "__main__":
    main()
