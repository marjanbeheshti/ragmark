"""
ZSInvert laundering test on RagMark watermarked vectors.

Literature attack: ZSInvert, "Universal Zero-shot Embedding Inversion"
(Zhang et al., 2025, arXiv:2504.00147), code: collinzrj/adversarial_decoding.

For N watermarked SQuAD vectors (same pipeline as validation.py):
  1. invert the WATERMARKED vector to text with ZSInvert, using
     all-mpnet-base-v2 as the target encoder (black-box queries only);
  2. re-embed the recovered text -> "laundered" vector;
  3. measure
       - text recovery      : BLEU vs. original passage, cosine to clean embedding
       - watermark survival : E_I verdict / confidence / source on laundered vector
       - retrieval utility  : top-5 hit rate of the doc's SQuAD questions,
                              original vs. laundered vector swapped into the index
       - strict mode        : hardened verdict on the laundered vector (no metadata)

Run from ~/regmark/submission in the zsinvert conda env:
  python experiments/zsinvert_attack.py --n 2 --max_steps 8     # smoke test
  python experiments/zsinvert_attack.py --n 50                  # full run
Writes results/zsinvert_attack.jsonl and results/zsinvert_attack_summary.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SUB = os.path.expanduser("~/regmark/submission")
AD = os.path.expanduser("~/Marjan/Local_Datasets/RAG_Mark/lit_attacks/adversarial_decoding")
sys.path[:0] = [SUB, os.path.join(SUB, "experiments"), AD]
os.chdir(SUB)

import ragmark as R  # noqa: E402
from validation import load_corpus, fit, KEY  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # noqa: E402
from adversarial_decoding.strategies.emb_inv_decoding import EmbInvDecoding  # noqa: E402

try:
    import ragmark_hardened as RH  # noqa: E402
except Exception as e:  # keep running even if the hardened module can't load here
    print("  (ragmark_hardened not importable:", e, ")")
    RH = None


def cos(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def questions_by_context(ds, texts):
    """Map each corpus passage to its SQuAD questions (empty if ds has no rows)."""
    want = set(texts)
    out = {}
    try:
        for row in ds:
            if isinstance(row, dict) and row.get("context") in want:
                out.setdefault(row["context"], []).append(row["question"])
    except Exception as e:
        print("  (could not read questions from ds:", e, ")")
    return out


def top5_hit(Q, DB, i):
    S = Q @ DB.T
    return float(np.mean([i in np.argsort(-s)[:5] for s in S]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of watermarked vectors to invert")
    ap.add_argument("--beam_width", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=32, help="max tokens of recovered text")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("ZSInvert laundering test (all-mpnet-base-v2)")
    print("=" * 60)

    # --- RagMark pipeline, identical to validation.py ---
    texts, ds = load_corpus(500)
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    E, U, mu = fit(texts, model, 768)
    d = E.shape[1]
    e1b, e0b, e0p = R.compute_bands(d)
    wm = np.array([R.watermark(E[i], i % 8, U, mu, e1b, e0b, e0p, doc_id=i)[0]
                   for i in range(len(E))])
    DB = wm / np.linalg.norm(wm, axis=1, keepdims=True)
    qmap = questions_by_context(ds, texts)
    print(f"  d={d}  E_I band={e0b}  ppb={e0p}  docs with questions={len(qmap)}")

    targets = [int(i) for i in rng.choice(len(wm), size=args.n, replace=False)]

    # --- ZSInvert, unchanged except the target is our vector, not a string ---
    attack = EmbInvDecoding(encoder=model, device="cuda",
                            should_natural=False, repetition_penalty=1.5)
    smooth = SmoothingFunction().method1

    out_path = "results/zsinvert_attack.jsonl"
    open(out_path, "w").close()
    rows = []

    for k, i in enumerate(targets):
        t0 = time.time()
        w = wm[i]
        _, c_w, verdict_w = R.decode_e0((w - mu) @ U.T, KEY, e0b, e0p)

        target = torch.tensor(w, dtype=torch.float32, device="cuda").unsqueeze(0)
        best = attack.run_decoding(prompt="tell me a story", target=target,
                                   beam_width=args.beam_width, max_steps=args.max_steps,
                                   top_k=args.top_k, top_p=1, should_full_sent=False,
                                   verbose=False, randomness=True)
        rec = best.seq_str

        v2 = model.encode(rec)
        s2, c2, verdict2 = R.decode_e0((v2 - mu) @ U.T, KEY, e0b, e0p)

        qs = qmap.get(texts[i], [])
        hit_o = hit_l = None
        if qs:
            Q = model.encode(qs, normalize_embeddings=True)
            hit_o = top5_hit(Q, DB, i)
            DBl = DB.copy()
            DBl[i] = v2 / np.linalg.norm(v2)
            hit_l = top5_hit(Q, DBl, i)

        strict = None
        if RH is not None:
            try:
                strict = str(RH.verify_hardened(v2, U, mu, e0b, e0p, meta=None))
            except Exception as e:
                strict = f"error: {e}"

        row = dict(
            idx=i, source=i % 8,
            original=texts[i][:300], recovered=rec,
            bleu=float(sentence_bleu([texts[i].split()], rec.split(),
                                     smoothing_function=smooth)),
            cos_rec_vs_clean=cos(v2, E[i]), cos_rec_vs_wm=cos(v2, w),
            cos_attack_reported=float(best.cos_sim or 0.0),
            wm_target_cI=float(c_w), wm_target_verdict=str(verdict_w),
            laundered_cI=float(c2), laundered_verdict=str(verdict2),
            laundered_source=None if s2 is None else int(s2),
            n_questions=len(qs),
            top5_hit_original=hit_o, top5_hit_laundered=hit_l,
            strict_verdict=strict,
            seconds=round(time.time() - t0, 1),
        )
        rows.append(row)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{k + 1}/{len(targets)}] doc {i}: cos(clean)={row['cos_rec_vs_clean']:.3f} "
              f"bleu={row['bleu']:.3f} laundered E_I={row['laundered_verdict']} "
              f"c_I={row['laundered_cI']:.3f} hit5 {hit_o}->{hit_l} ({row['seconds']}s)")
        if k < 3:
            print("    original :", texts[i][:160].replace("\n", " "))
            print("    recovered:", rec[:160].replace("\n", " "))

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = dict(
        attack="ZSInvert (Zhang et al. 2025, arXiv:2504.00147)",
        encoder="all-mpnet-base-v2", n=len(rows), params=vars(args),
        watermarked_targets_pass_EI=sum(r["wm_target_verdict"] == "PASS" for r in rows),
        laundered_pass_EI=sum(r["laundered_verdict"] == "PASS" for r in rows),
        laundered_pass_with_correct_source=sum(
            r["laundered_verdict"] == "PASS" and r["laundered_source"] == r["source"] for r in rows),
        mean_laundered_cI=mean("laundered_cI"),
        mean_bleu=mean("bleu"),
        mean_cos_rec_vs_clean=mean("cos_rec_vs_clean"),
        mean_cos_rec_vs_wm=mean("cos_rec_vs_wm"),
        top5_hit_original=mean("top5_hit_original"),
        top5_hit_laundered=mean("top5_hit_laundered"),
        strict_verdicts=sorted({str(r["strict_verdict"]) for r in rows}),
    )
    with open("results/zsinvert_attack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))
    print("Saved results/zsinvert_attack.jsonl and results/zsinvert_attack_summary.json")


if __name__ == "__main__":
    main()
