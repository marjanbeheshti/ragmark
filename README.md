# RAGMARK

Hybrid embedding watermark for RAG vector databases: simultaneous
**tamper detection** and **source attribution** at the embedding layer.

RAGMARK watermarks each embedding during ingestion and verifies it at
retrieval time, before the passage reaches the LLM. It combines:

- **Attribution Layer (E_A)** — greedy pool-based selection in PCA band
  `[round(0.04d), round(0.2d))`. Encodes an 8-bit source ID by choosing
  naturally-agreeing coordinate pairs; near-zero distortion (6 of 12,000
  pairs adjusted at d=768). The selection is content-dependent and cannot
  be re-derived, so it is stored and HMAC-signed.
- **Integrity Layer (E_I)** — swap-mode in band `[round(0.2d), round(0.5d))`
  with a self-derived **full-vector** fingerprint (hashes all d PC
  coordinates). Detects modification of any coordinate; defeats
  confidence-oracle hill-climbing.

Metadata is authenticated with HMAC-SHA-256 under an independent key
`K' = SHA-256(K || "|hmac")`, binding `(f_A, f_I, doc_id, source_id, sel)`.

## Single source of truth

All watermarking logic lives in **`ragmark.py`**. Every experiment script
imports it — there is no duplicated implementation. Numbers in the paper
are produced by running the scripts in `experiments/` against this core.

## Requirements

```
python >= 3.11
numpy, scipy, scikit-learn
sentence-transformers, datasets   # embeddings + corpora
```
Install: `pip install numpy scipy scikit-learn sentence-transformers datasets`

The LLM trick-rate experiment additionally needs **Ollama** running
locally with the target model (default `qwen3:8b`).

Datasets (SQuAD, BEIR SciFact) download automatically via `datasets`.
Attack corpora: the Black-Hole reference implementation and the
PoisonedRAG adversarial passages are expected under the paths named in
`e2e_attacks.py` (see that file's header).

## Repository structure

```
ragmark.py              Core: embed / verify / HMAC (single source of truth)
experiments/
  validation.py         CosSim, self-check, GT recall, centroid/random
                        rejection, content-binding ablation, guessed-key
                        forgery, confidence oracle, HMAC sel-tamper,
                        cross-model (384/768/1024), latency
  e2e_attacks.py        Black-Hole end-to-end (5 seeds) + PoisonedRAG
                        rejection + PoisonedRAG LLM trick rate (--with-llm)
  supplementary.py      KS invisibility, retrieval utility (Spearman,
                        NDCG, rank shift), 10k-doc scale recall
  noise_ablation.py     E_A in-band noise sweep + oracle error bars
  security_tests.py     E_I-band modification attack (single- vs dual-
                        layer), storage transforms, threshold sweep,
                        E_I in-band noise trichotomy
  domain_separation.py  Domain-separation revalidation: tagged vs untagged
                        seeds (stream-overlap 0/100), self-check, KS
  metadata_replay.py    Metadata-replay / cross-source-swap: E_A-only vs
                        dual-layer, s_A!=s_I cross-check value
results/                JSON outputs written by each script (one per script)
paper/                  RAGMARK.tex, references.bib
```

## Reproducing the paper

```bash
# core validation (all headline numbers)
python experiments/validation.py

# end-to-end attacks (add --with-llm for the trick-rate table)
python experiments/e2e_attacks.py --with-llm

# invisibility, utility, scale
python experiments/supplementary.py

# E_A noise robustness + oracle CIs
python experiments/noise_ablation.py

# E_I-band attack, storage, threshold sweep, E_I noise trichotomy
python experiments/security_tests.py

# domain-separation revalidation (tagged vs untagged seeds)
python experiments/domain_separation.py

# metadata-replay / cross-source-swap
python experiments/metadata_replay.py
```

## Paper table / claim → script → result JSON

Every number in the paper is produced by one of the seven scripts below and
serialized to a JSON file in `results/`. This table maps each paper table (or
headline claim) to the script that produces it and the exact JSON file that
holds the reproduced numbers.

| Paper location | Claim | Script | Result JSON |
|---|---|---|---|
| §V-A, Table IX row | CosSim 0.976 (d=768) | `validation.py` | `results/validation.json` |
| §V-A | E_A 500/500, E_I 499/500 self-check | `validation.py` | `results/validation.json` |
| Table IX | GT recall parity (Recall@5 .894→.892) | `validation.py` | `results/validation.json` |
| §V-B | centroid / random rejection 200/200 | `validation.py` | `results/validation.json` |
| §V-B, ablation | content-binding ablation (200 bound vs 110 unbound) | `validation.py` | `results/validation.json` |
| Prop. 2 / §V-C | guessed-key forgery 0/1000 | `validation.py` | `results/validation.json` |
| Table VI | confidence oracle 0/50 (max 0.800) | `validation.py` (+ CIs in `noise_ablation.py`) | `results/validation.json`, `results/noise_ablation.json` |
| §IV-D | HMAC sel-tamper 500/500 caught | `validation.py` | `results/validation.json` |
| Table XII | cross-model 0.914 (d=384) / 0.984 (d=1024) | `validation.py` | `results/validation.json` |
| Table (Black-Hole) | Black-Hole 1050/1050, recall recovery | `e2e_attacks.py` | `results/e2e_attacks.json` |
| Table VII / §V-E | PoisonedRAG 500/500 rej (max conf ≤0.775) | `e2e_attacks.py` | `results/e2e_attacks.json` |
| §V (LLM) | PoisonedRAG LLM trick rate 95% → 2% | `e2e_attacks.py --with-llm` | `results/e2e_attacks.json` |
| Table VIII | KS invisibility p=1.0 all regions | `supplementary.py` | `results/supplementary.json` |
| Table IX | Spearman ρ 0.995, 88.2% docs unchanged, NDCG@5 | `supplementary.py` | `results/supplementary.json` |
| Table XII (scale) | 10k-doc scale recall gap ~0.2 pt | `supplementary.py` | `results/supplementary.json` |
| §III (bands) | 6/12000 forced pairs, Spearman gap, KS | `supplementary.py` | `results/supplementary.json` |
| Table XI | E_A in-band noise sweep + dual-layer coverage | `noise_ablation.py` | `results/noise_ablation.json` |
| Table X | E_I in-band noise trichotomy (decode/detect=500) | `security_tests.py` | `results/security_tests.json` |
| §V-C / §V-E | E_I-band attack: E_A-only 500/500 vuln, dual 0/500 | `security_tests.py` | `results/security_tests.json` |
| Table XIII | storage transforms (float16/bfloat16/int8) | `security_tests.py` | `results/security_tests.json` |
| Table VII | threshold sweep (FPR=0 for τ∈[0.77,0.99]) | `security_tests.py` | `results/security_tests.json` |
| §IV-B / domain sep. | stream-overlap 0/100, tagged≠untagged seeds | `domain_separation.py` | `results/domain_separation.json` |
| §IV-C / metadata replay | cross-source swap: E_A-only 11/500 wrong, dual 500/500, s_A≠s_I sole in 11 | `metadata_replay.py` | `results/metadata_replay.json` |

### results/ directory

```
results/
  validation.json          <- validation.py
  e2e_attacks.json         <- e2e_attacks.py
  supplementary.json       <- supplementary.py
  noise_ablation.json      <- noise_ablation.py
  security_tests.json      <- security_tests.py
  domain_separation.json   <- domain_separation.py
  metadata_replay.json     <- metadata_replay.py
```

### Paper numbers pending reconciliation with shipped JSONs

The current `.tex` draft still carries a few pre-consolidation numbers that
differ from what the shipped code now produces. Update the paper to match the
JSONs before final submission:

| Paper says | Shipped JSON says | Where |
|---|---|---|
| ablation 157/200 | 110/200 (`validation.json`) | §V-B |
| oracle max 0.775 | 0.800 (`validation.json`) | Table VI |
| PoisonedRAG max conf 0.762 | 0.775 (`e2e_attacks.json`) | Table VII, §V-E |
| storage float16 374/500 | 381/500 (`security_tests.json`) | Table XIII |
| storage bfloat16 11/500 | 9/500 (`security_tests.json`) | Table XIII |
| storage int8 0/500 | 1/500 (`security_tests.json`) | Table XIII |
| `K' = KDF(K, "hmac")` | `K' = SHA-256(K ‖ "|hmac")` | §IV (Keys) |

## Security notes

- The **PCA basis (U, μ) is public**, derived from the corpus. Security
  rests solely on the secret key `K`: pair positions come from
  `SHA-256(K || tag || f)` (domain tags `"attr"` / `"integ"`), uniform to
  an attacker without `K`.
- The content fingerprint `f_A` is public and carries no secrecy; its role
  is content-binding (defeats centroid-averaging: 200/200 rejected vs
  110/200 without binding).
- Pair derivation uses NumPy's `default_rng` (PCG64); reproduction requires
  the same generator.
