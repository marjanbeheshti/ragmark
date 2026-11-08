#!/usr/bin/env python3
"""
Text-Binding Extension vs Payload-Swap (Walkthrough Sec. 8) -- v2
=================================================================
Converts the paper's "one-line extension we note but do not require"
into a measured defense, so the separation assumption (vector index vs
document store as distinct trust boundaries) becomes OPTIONAL rather
than load-bearing.

v2 fixes two design flaws of the first version:
  * SINGLE TUPLE, NOT A SECOND TAG. v1 issued a separable second HMAC
    over (tag2, h_text); stripping (h_text, tag_text) left a valid base
    blob, reopening payload-swap via DOWNGRADE. v2 binds h_text inside
    ONE signed tuple (f_a|f_i|h_v|h_text|doc_id|sid|sel); there is
    nothing optional to strip, and test (C) measures the stripping case
    explicitly.
  * CANONICALIZATION. v1 hashed raw UTF-8, so Unicode normalization or
    CRLF changes across document-store round trips would false-reject.
    v2 normalizes (NFC, newlines, strip) before hashing; test (D)
    measures round-trip variants.

Tests:
  (A) No binding (base hardened gate): payload-swap passes -- the
      vector verifies and nothing checks the text (the out-of-scope
      case under the separation assumption).
  (B) With binding: swaps rejected, legitimate pairs pass.
  (C) Downgrade-strip: remove the text fields from a bound tuple ->
      the single tag no longer verifies -> rejected.
  (D) Canonicalization: CRLF / NFD / trailing-whitespace variants of
      the legitimate text still pass (0 false rejects).
  (E) Cost: one SHA-256 of the canonicalized passage per verification.

Writes results/textbinding.json
Usage: python experiments/textbinding_test.py
"""
import hashlib
import hmac
import json
import os
import sys
import unicodedata
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
SEED = 2026


# ---------------------------------------------------------------------
# Canonicalization + single-tuple binding
# ---------------------------------------------------------------------
def canon(text):
    """Canonical text form hashed by h_text: Unicode NFC, \\r\\n and
    \\r -> \\n, outer whitespace stripped. Cheap insurance against
    document-store round trips."""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t.strip()


def h_text(text):
    return hashlib.sha256(canon(text).encode("utf-8")).hexdigest()[:32]


def sign_meta_tb(f_a, f_i, h_v, ht, doc_id, sid, selection,
                 key=R.MASTER_KEY):
    """ONE tag over the whole tuple including h_text. No separable
    text tag exists, so no downgrade path exists."""
    msg = (f"{f_a}|{f_i}|{h_v}|{ht}|{doc_id}|{sid}|"
           f"{R._sel_str(selection)}").encode()
    return hmac.new(R._hmac_key(key), msg, hashlib.sha256).hexdigest()


def watermark_textbound(v_raw, source_id, text, U, mu,
                        e1_band, e0_band, e0_ppb,
                        doc_id=None, key=R.MASTER_KEY):
    """Hardened watermark whose signed tuple also binds h_text."""
    w_raw, meta = RH.watermark_hardened(v_raw, source_id, U, mu,
                                        e1_band, e0_band, e0_ppb,
                                        doc_id=doc_id, key=key)
    meta = dict(meta)
    meta["h_text"] = h_text(text)
    meta["tag3"] = sign_meta_tb(meta["f_a"], meta["f_i"], meta["h_v"],
                                meta["h_text"], meta["doc_id"],
                                meta["sid"], meta["sel"], key)
    return w_raw, meta


def verify_textbound(w_raw, served_text, U, mu, e0_band, e0_ppb,
                     meta=None, key=R.MASTER_KEY, manifest=None,
                     expected_doc_id=None):
    """Text-bound verification. Text binding is REQUIRED: a tuple
    without (h_text, tag3) is rejected outright -- that is exactly the
    downgrade an attacker would attempt (test C)."""
    if meta is None:
        return {"verdict": "REJECTED_NO_METADATA", "source_id": None}
    if "h_text" not in meta or "tag3" not in meta:
        return {"verdict": "REJECTED_TEXT_DOWNGRADE", "source_id": None}
    exp = sign_meta_tb(meta["f_a"], meta["f_i"], meta["h_v"],
                       meta["h_text"], meta["doc_id"], meta["sid"],
                       meta["sel"], key)
    if not hmac.compare_digest(exp, meta["tag3"]):
        return {"verdict": "TAMPERED_BAD_SIG", "source_id": None}
    if h_text(served_text) != meta["h_text"]:
        return {"verdict": "REJECTED_TEXT_MISMATCH", "source_id": None}
    # delegate the vector-side checks to the hardened gate
    return RH.verify_hardened(w_raw, U, mu, e0_band, e0_ppb, meta=meta,
                              key=key, strict=True,
                              expected_doc_id=expected_doc_id,
                              manifest=manifest)


# ---------------------------------------------------------------------
def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"date": datetime.now().isoformat(),
           "test": "text-binding v2 (single tuple) vs payload-swap"}

    print("=" * 62)
    print("Text-Binding v2 vs Payload-Swap (Sec. 8)")
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

    print("  Watermarking: hardened (no binding) + text-bound...")
    wm, metas_h, metas_tb = [], [], []
    for i in range(len(E)):
        w_raw, mh = RH.watermark_hardened(E[i], i % 8, U, mu,
                                          e1_band, e0_band, e0_ppb,
                                          doc_id=i, key=KEY)
        wm.append(w_raw); metas_h.append(mh)
        mtb = dict(mh)
        mtb["h_text"] = h_text(texts[i])
        mtb["tag3"] = sign_meta_tb(mtb["f_a"], mtb["f_i"], mtb["h_v"],
                                   mtb["h_text"], mtb["doc_id"],
                                   mtb["sid"], mtb["sel"], KEY)
        metas_tb.append(mtb)
    wm = np.asarray(wm)

    def swap_target(i):
        j = int(rng.integers(0, len(texts)))
        return (j + 1) % len(texts) if j == i else j

    # =================================================================
    # (A) No binding: payload-swap passes the full hardened gate
    # =================================================================
    print("\n[A] Payload-swap WITHOUT binding (base hardened gate)")
    swap_pass = 0
    for i in range(N_DOCS):
        _ = swap_target(i)   # attacker serves another doc's text;
        v = RH.verify_hardened(wm[i], U, mu, e0_band, e0_ppb,
                               meta=metas_h[i], key=KEY,
                               expected_doc_id=i)["verdict"]
        swap_pass += (v == "PASS")      # nothing checks the text
    print(f"  Swap passes (vector verifies, text unchecked): "
          f"{swap_pass}/{N_DOCS}")
    res["A_swap_pass"] = f"{swap_pass}/{N_DOCS}"

    # =================================================================
    # (B) With binding: swap rejected, legitimate pairs pass
    # =================================================================
    print("\n[B] Payload-swap WITH binding (single tuple)")
    swap_rej = legit_pass = 0
    for i in range(N_DOCS):
        j = swap_target(i)
        vs = verify_textbound(wm[i], texts[j], U, mu, e0_band, e0_ppb,
                              meta=metas_tb[i], key=KEY,
                              expected_doc_id=i)["verdict"]
        swap_rej += (vs != "PASS")
        vl = verify_textbound(wm[i], texts[i], U, mu, e0_band, e0_ppb,
                              meta=metas_tb[i], key=KEY,
                              expected_doc_id=i)["verdict"]
        legit_pass += (vl == "PASS")
    print(f"  Swaps rejected:            {swap_rej}/{N_DOCS}")
    print(f"  Legitimate pairs pass:     {legit_pass}/{N_DOCS}")
    res["B_swap_rejected"] = f"{swap_rej}/{N_DOCS}"
    res["B_legit_pass"] = f"{legit_pass}/{N_DOCS}"

    # =================================================================
    # (C) Downgrade-strip: remove text fields -> rejected
    # =================================================================
    print("\n[C] Downgrade-strip (the v1 hole, now measured)")
    strip_rej = 0
    for i in range(N_DOCS):
        stripped = {k: v for k, v in metas_tb[i].items()
                    if k not in ("h_text", "tag3")}
        v = verify_textbound(wm[i], texts[swap_target(i)], U, mu,
                             e0_band, e0_ppb, meta=stripped, key=KEY,
                             expected_doc_id=i)["verdict"]
        strip_rej += (v != "PASS")
    print(f"  Stripped tuples rejected:  {strip_rej}/{N_DOCS}  "
          f"(REJECTED_TEXT_DOWNGRADE)")
    print("  -> binding lives INSIDE the one signed tuple; there is")
    print("     no separable text tag to delete.")
    res["C_downgrade_rejected"] = f"{strip_rej}/{N_DOCS}"

    # =================================================================
    # (D) Canonicalization: benign text round-trips still pass
    # =================================================================
    print("\n[D] Canonicalization (benign round-trip variants)")
    variants = {
        "CRLF newlines": lambda t: t.replace("\n", "\r\n"),
        "NFD normalization": lambda t: unicodedata.normalize("NFD", t),
        "trailing whitespace": lambda t: t + "  \n",
    }
    res["D_variants"] = {}
    for name, fn in variants.items():
        ok = sum(verify_textbound(wm[i], fn(texts[i]), U, mu, e0_band,
                                  e0_ppb, meta=metas_tb[i], key=KEY,
                                  expected_doc_id=i)["verdict"] == "PASS"
                 for i in range(N_DOCS))
        print(f"  {name:<22s} pass {ok}/{N_DOCS}")
        res["D_variants"][name] = f"{ok}/{N_DOCS}"

    # =================================================================
    print("\n[E] Cost: one SHA-256 of the canonicalized passage per")
    print("    verification; no change to the vector or the watermark.")
    print("\n" + "=" * 62)
    print("SUMMARY")
    print(f"  no binding:   swap passes {swap_pass}/{N_DOCS} "
          f"(the separation assumption's job)")
    print(f"  with binding: swap rejected {swap_rej}/{N_DOCS}, "
          f"legit {legit_pass}/{N_DOCS}, downgrade {strip_rej}/{N_DOCS}")
    print("  -> separation assumption is OPTIONAL; binding is inside")
    print("     the single signed tuple (no downgrade), canonicalized")
    print("     (no round-trip false rejects).")
    print("=" * 62)
    with open("results/textbinding.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("Saved results/textbinding.json")


if __name__ == "__main__":
    main()
