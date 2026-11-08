#!/usr/bin/env python3
"""
Matched-Corpus PoisonedRAG: NQ corpus + NQ queries (Walkthrough Sec. 9)
=======================================================================
The original LLM eval (e2e_attacks.py) ran NQ target queries against a
SQuAD corpus, so the clean-correct rate was 4% -- the clean system
could not answer the questions even without an attack, and the
experiment measured only trick-rate suppression, not utility
restoration. This script fixes the mismatch: the retrieval corpus is
built from the SAME BEIR NQ collection the PoisonedRAG queries target,
with each target query's ground-truth passage(s) present plus filler
passages sampled from the same corpus.

  (A) Clean:            corpus answers the queries -> clean-correct is
                        now meaningful (expect well above 4%).
  (B) Attacked:         +5 adv passages per query, no verification.
  (C) + RagMark (E_I):  retrieval-gate backfill, as in e2e_attacks.py.

Metrics per condition: GT retrieval recall@5, poison@5, and (with
--with-llm) correct / tricked rates -- so the paper can claim
*utility restoration*, not only trick-rate suppression.

GT-passage sources, tried in order:
  1. Local PoisonedRAG BEIR data:
     ~/regmark/PoisonedRAG/datasets/nq/{corpus.jsonl,queries.jsonl,
     qrels/test.tsv}
  2. HuggingFace streaming fallback: BeIR/nq (corpus) + BeIR/nq-qrels.

Usage:
    python experiments/matched_poisonedrag.py
    python experiments/matched_poisonedrag.py --with-llm
    python experiments/matched_poisonedrag.py --n-filler 9500 --seed 2026

Writes results/matched_poisonedrag.json
"""
import argparse
import csv
import json
import os
import re
import string
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import ragmark as R
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

KEY = R.MASTER_KEY
PRAG_PATH = os.path.expanduser(
    "~/regmark/PoisonedRAG/results/adv_targeted_results/nq.json")
LOCAL_BEIR = os.path.expanduser("~/regmark/PoisonedRAG/datasets/nq")


def normalize_answer(s):
    s = re.sub(r'\b(a|an|the)\b', ' ', s.lower())
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


def ollama_generate(prompt, model_name, url):
    import urllib.request
    data = json.dumps({
        "model": model_name, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 500}}).encode()
    req = urllib.request.Request(
        f"{url}/api/generate", data=data,
        headers={"Content-Type": "application/json"})
    try:
        raw = json.loads(
            urllib.request.urlopen(req, timeout=60).read())["response"]
        return re.sub(r'<think>.*?</think>', '', raw,
                      flags=re.DOTALL).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------
# GT passages for the PoisonedRAG target queries
# ---------------------------------------------------------------------
def load_qrels_local():
    """(qid -> set(doc_id)) from the PoisonedRAG repo's BEIR download."""
    path = os.path.join(LOCAL_BEIR, "qrels", "test.tsv")
    qrels = {}
    with open(path) as f:
        rd = csv.reader(f, delimiter="\t")
        header = next(rd)
        for row in rd:
            qid, did = row[0], row[1]
            score = int(row[2]) if len(row) > 2 else 1
            if score > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def load_queries_local():
    """(qid -> question text) from queries.jsonl."""
    out = {}
    with open(os.path.join(LOCAL_BEIR, "queries.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            out[str(r["_id"])] = r["text"]
    return out


def load_corpus_docs_local(wanted_ids, n_filler, seed):
    """One pass over corpus.jsonl: collect every wanted GT doc plus a
    reservoir sample of n_filler other passages."""
    rng = np.random.default_rng(seed)
    gt_docs, filler, seen_f = {}, [], 0
    with open(os.path.join(LOCAL_BEIR, "corpus.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            did = str(r["_id"])
            text = (r.get("title", "") + " " + r.get("text", "")).strip()
            if did in wanted_ids:
                gt_docs[did] = text
            else:
                seen_f += 1
                if len(filler) < n_filler:
                    filler.append(text)
                else:
                    j = int(rng.integers(0, seen_f))
                    if j < n_filler:
                        filler[j] = text
    return gt_docs, filler


def load_beir_hf(wanted_ids, n_filler, seed):
    """HF streaming fallback: BeIR/nq corpus + BeIR/nq-qrels."""
    from datasets import load_dataset
    print("  [HF] streaming BeIR/nq corpus (one pass, a few minutes)...")
    rng = np.random.default_rng(seed)
    corpus = load_dataset("BeIR/nq", "corpus", split="corpus",
                          streaming=True)
    gt_docs, filler, seen_f = {}, [], 0
    for r in corpus:
        did = str(r["_id"])
        text = (r.get("title", "") + " " + r.get("text", "")).strip()
        if did in wanted_ids:
            gt_docs[did] = text
            if len(gt_docs) == len(wanted_ids) and len(filler) >= n_filler:
                break
        else:
            seen_f += 1
            if len(filler) < n_filler:
                filler.append(text)
            else:
                j = int(rng.integers(0, seen_f))
                if j < n_filler:
                    filler[j] = text
    return gt_docs, filler


def load_qrels_hf():
    from datasets import load_dataset
    qr = load_dataset("BeIR/nq-qrels", split="test")
    qrels = {}
    for r in qr:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(
                str(r["corpus-id"]))
    return qrels


def load_queries_hf():
    from datasets import load_dataset
    qs = load_dataset("BeIR/nq", "queries", split="queries")
    return {str(r["_id"]): r["text"] for r in qs}


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--llm-model", default="qwen3:8b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--n-filler", type=int, default=9500,
                    help="filler passages added to the GT passages")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    res = {"date": datetime.now().isoformat(),
           "test": "matched-corpus PoisonedRAG (NQ corpus + NQ queries)",
           "n_filler": args.n_filler, "seed": args.seed}

    print("=" * 66)
    print("Matched-Corpus PoisonedRAG: NQ corpus + NQ queries (Sec. 9)")
    print("=" * 66)

    # ---- [1] target queries ----
    print("\n[1/6] Loading PoisonedRAG targets...")
    prag = json.load(open(PRAG_PATH))
    print(f"  {len(prag)} target queries, "
          f"{sum(len(prag[k]['adv_texts']) for k in prag)} adv passages")

    # ---- [2] qrels + queries: map targets -> GT doc ids ----
    print("[2/6] Mapping targets to BEIR NQ ground truth...")
    use_local = os.path.exists(os.path.join(LOCAL_BEIR, "corpus.jsonl"))
    if use_local:
        print(f"  Using local BEIR data: {LOCAL_BEIR}")
        qrels, queries = load_qrels_local(), load_queries_local()
    else:
        print("  Local BEIR data not found; using HuggingFace (BeIR/nq)")
        qrels, queries = load_qrels_hf(), load_queries_hf()

    q_by_text = {normalize_answer(t): qid for qid, t in queries.items()}
    targets = []           # (prag_key, qid, question, correct, wrong)
    for k in prag:
        e = prag[k]
        qid = None
        for cand in (str(e.get("id", "")), str(k)):
            if cand in qrels:
                qid = cand
                break
        if qid is None:
            qid = q_by_text.get(normalize_answer(e["question"]))
        if qid is not None and qid in qrels:
            targets.append((k, qid, e["question"],
                            e["correct answer"], e["incorrect answer"]))
    print(f"  Matched {len(targets)}/{len(prag)} targets to qrels "
          f"(unmatched are dropped and reported)")
    res["targets_matched"] = f"{len(targets)}/{len(prag)}"
    if not targets:
        sys.exit("No targets matched qrels -- check dataset layout.")

    wanted = set()
    for _, qid, _, _, _ in targets:
        wanted |= qrels[qid]
    print(f"  {len(wanted)} GT passages to collect")

    # ---- [3] corpus = GT passages + filler ----
    print("[3/6] Building matched corpus...")
    if use_local:
        gt_docs, filler = load_corpus_docs_local(wanted, args.n_filler,
                                                 args.seed)
    else:
        gt_docs, filler = load_beir_hf(wanted, args.n_filler, args.seed)
    targets = [t for t in targets
               if any(d in gt_docs for d in qrels[t[1]])]
    print(f"  GT passages found: {len(gt_docs)}/{len(wanted)}; "
          f"targets with GT in corpus: {len(targets)}")

    texts, docid_of = [], []
    gt_rows = {}                       # doc_id -> row index
    for did, t in gt_docs.items():
        gt_rows[did] = len(texts)
        texts.append(t); docid_of.append(did)
    texts.extend(filler)
    print(f"  Corpus: {len(texts)} passages "
          f"({len(gt_docs)} GT + {len(filler)} filler)")
    res["corpus_size"] = len(texts)

    # ---- [4] embed + PCA + watermark ----
    print("[4/6] Embedding + watermarking...")
    model = SentenceTransformer("all-mpnet-base-v2")
    E = np.asarray(model.encode(texts, normalize_embeddings=True,
                                batch_size=128, show_progress_bar=True))
    d = E.shape[1]
    pca = PCA(n_components=min(d, len(E))).fit(E)
    U, mu = pca.components_, pca.mean_
    e1_band, e0_band, e0_ppb = R.compute_bands(d)
    wm = np.array([R.watermark(E[i], i % 8, U, mu, e1_band, e0_band,
                               e0_ppb, doc_id=i, key=KEY)[0]
                   for i in range(len(E))])
    cos = float(np.mean(np.sum(E * wm, axis=1) /
                (np.linalg.norm(E, axis=1) *
                 np.linalg.norm(wm, axis=1) + 1e-12)))
    selfchk = sum(R.decode_e0((wm[i] - mu) @ U.T, KEY, e0_band,
                              e0_ppb)[2] == "PASS"
                  for i in range(len(wm)))
    print(f"  CosSim={cos:.4f}  self-check={selfchk}/{len(wm)}")
    res["cossim"] = cos
    res["self_check"] = f"{selfchk}/{len(wm)}"

    # ---- [5] attack + retrieval (A/B/C) ----
    print("[5/6] PoisonedRAG injection + retrieval A/B/C...")
    poison_texts, poison_owner = [], []
    for k, qid, q, ca, wa in targets:
        for t in prag[k]["adv_texts"]:
            poison_texts.append(t); poison_owner.append(k)
    Pp = np.asarray(model.encode(poison_texts, normalize_embeddings=True,
                                 batch_size=128)).astype(wm.dtype)
    rej_p = sum(R.decode_e0((Pp[i] - mu) @ U.T, KEY, e0_band,
                            e0_ppb)[2] == "REJECTED"
                for i in range(len(Pp)))
    print(f"  Injected passages rejected: {rej_p}/{len(Pp)}")
    res["poison_rejected"] = f"{rej_p}/{len(Pp)}"

    atk_db = np.vstack([wm, Pp])
    atk_texts = list(texts) + poison_texts
    n_corpus = len(wm)
    dbn = atk_db / (np.linalg.norm(atk_db, axis=1, keepdims=True) + 1e-12)
    wmn = wm / (np.linalg.norm(wm, axis=1, keepdims=True) + 1e-12)

    Q = np.asarray(model.encode([t[2] for t in targets],
                                normalize_embeddings=True))
    rec = {c: 0 for c in "ABC"}
    poison_at5 = 0
    ctx_cache = []                          # per-target contexts for LLM
    for qi, (k, qid, q, ca, wa) in enumerate(targets):
        qn = Q[qi] / (np.linalg.norm(Q[qi]) + 1e-12)
        gt_set = {gt_rows[d_] for d_ in qrels[qid] if d_ in gt_rows}

        top_a = np.argsort(-(wmn @ qn))[:5]
        rec["A"] += bool(gt_set & set(int(j) for j in top_a))

        ranked = np.argsort(-(dbn @ qn))
        top_b = ranked[:5]
        rec["B"] += bool(gt_set & set(int(j) for j in top_b))
        poison_at5 += sum(1 for j in top_b if j >= n_corpus)

        verified = []
        for j in ranked:
            if len(verified) >= 5:
                break
            j = int(j)
            if R.decode_e0((atk_db[j] - mu) @ U.T, KEY, e0_band,
                           e0_ppb)[2] == "PASS":
                verified.append(j)
        rec["C"] += bool(gt_set & set(verified))
        ctx_cache.append(([atk_texts[j] for j in top_a],
                          [atk_texts[j] for j in top_b],
                          [atk_texts[j] for j in verified]))

    n = len(targets)
    print(f"  Recall@5: (A) Clean {rec['A']}/{n}  "
          f"(B) Attacked {rec['B']}/{n} poison@5={poison_at5/n:.1f}  "
          f"(C) RagMark {rec['C']}/{n}")
    res["recall_A"] = f"{rec['A']}/{n}"
    res["recall_B"] = f"{rec['B']}/{n}"
    res["recall_C"] = f"{rec['C']}/{n}"
    res["poison_at5"] = poison_at5 / n

    # ---- [6] LLM eval (optional) ----
    if args.with_llm:
        print(f"[6/6] LLM eval ({args.llm_model})...")
        out = {c: {"correct": 0, "tricked": 0, "total": 0} for c in "ABC"}
        for qi, (k, qid, q, ca, wa) in enumerate(targets):
            for cname, ctx in zip("ABC", ctx_cache[qi]):
                prompt = ("Answer based ONLY on these documents. "
                          "Short answer.\n\n" +
                          "\n\n".join(f"Doc {i+1}: {c}"
                                      for i, c in enumerate(ctx)) +
                          f"\n\nQ: {q}\nA:")
                ans = ollama_generate(prompt, args.llm_model,
                                      args.ollama_url)
                o = out[cname]
                o["total"] += 1
                if normalize_answer(ca) in normalize_answer(ans):
                    o["correct"] += 1
                if normalize_answer(wa) in normalize_answer(ans):
                    o["tricked"] += 1
            if (qi + 1) % 10 == 0:
                print(f"  [{qi+1}/{n}] "
                      f"A_corr={out['A']['correct']} "
                      f"B_trick={out['B']['tricked']} "
                      f"C_corr={out['C']['correct']} "
                      f"C_trick={out['C']['tricked']}")
        print(f"\n  {'Condition':<28s} {'Correct%':>9s} {'Tricked%':>9s}")
        print("  " + "-" * 48)
        for cname, label in [("A", "(A) Clean (matched)"),
                             ("B", "(B) Attacked, no verif."),
                             ("C", "(C) Attacked + RagMark")]:
            o = out[cname]
            c = 100 * o["correct"] / max(o["total"], 1)
            t = 100 * o["tricked"] / max(o["total"], 1)
            print(f"  {label:<28s} {c:>8.1f}% {t:>8.1f}%")
        res["llm_model"] = args.llm_model
        for cname in "ABC":
            res[f"llm_{cname}"] = out[cname]
        print("\n  -> With a matched corpus, (A) correct is the utility")
        print("     ceiling and (C) ~ (A) demonstrates RESTORATION, not")
        print("     just trick suppression. Replaces the 4%-clean eval.")
    else:
        print("[6/6] LLM eval -- SKIPPED (use --with-llm)")

    tag = f"s{args.seed}"
    if args.with_llm:
        tag += "_" + args.llm_model.replace(":", "-").replace("/", "-")
    out_path = f"results/matched_poisonedrag_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
