# swiss-ruling-retrieval

Multilingual (DE/FR/IT) Swiss legal retrieval: given a Federal Supreme
Court case, retrieve the rulings (and law articles) it cites. See
"Results" below for measured results.

## Quick start

```bash
pip install -r requirements.txt                      # Python >= 3.11
python3 -m scripts.inspect_data --split=validation   # downloads ~35 MB, prints stats
python3 -m scripts.evaluate --train-cases 5000 --val-queries 500 --restrict-corpus
```

The last command is the fast dev-subset evaluation (BM25, chunked BM25, dense,
constant baseline); it runs on CPU. Measured numbers are under "Results", and
fine-tuning is under "Fine-tuning". Run the tests with `python3 -m pytest`.

## Setup

`pip install -r requirements.txt` installs everything with a platform-neutral
`torch` pin, so pip picks the build for your machine (CPU or a recent GPU; see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) if you need a
specific one). Retrieval/evaluation (not fine-tuning) runs fine on CPU.

For an old (Pascal-era, e.g. GTX 1060) CUDA GPU, install
`requirements-cu126.txt` instead: newer default torch wheels dropped that GPU's
architecture, so it pulls PyTorch's CUDA 12.6 build.

## Getting the data

Data isn't committed (`data/` is gitignored). It's public on the Hugging
Face hub - no account or token required -
[`rcds/swiss_doc2doc_ir`](https://huggingface.co/datasets/rcds/swiss_doc2doc_ir).

`scripts/inspect_data.py` downloads a split automatically if it's not
already present locally, then prints its schema and summary statistics:

```bash
python3 -m scripts.inspect_data --split=validation  # or: train, test
```

(run as a module, from the repo root, so it can import the shared
`DATA_DIR` constant from `paths.py` and the loader from `lib/dataset.py`)

Start with `validation` (~35 MB) - it's the smallest split and enough to
develop against. `train` (~243 MB) and `test` (~85 MB) use the same
interface.

Loading and parsing split files (downloading, schema, the `Case`
dataclass) lives in `lib/dataset.py`:
- `ensure_data_present(split=...)` - downloads a split if needed, writes
  one `<decision_id>.json` file per record under `data/data/<split>/`
  (for browsing a single case by hand), then deletes the downloaded
  `.jsonl.xz` - the per-record files are the canonical local copy from
  then on. Idempotent, runs once.
- `iter_cases(split=...)` - calls the above if needed, then returns
  every record as a `Case` in a list, sorted by `decision_id` (the
  whole split is held in memory; a progress bar shows the loading). Use
  this directly if you don't want to run the full inspection.

Known data quirk:
`laws`/`cited_rulings` need `ast.literal_eval`, not `json.loads`.
See "## License" below for this dataset's license.

The retrieval corpus for the primary task (rulings cited by `laws`) is
[`rcds/swiss_leading_decisions`](https://huggingface.co/datasets/rcds/swiss_leading_decisions)
(21,171 rulings, ~90.7% of `cited_rulings` resolve against it; the rest
are cited but absent from that dataset). `lib/rulings.py`'s
`ensure_rulings_data_present()` downloads it if needed and expands it
the same way: one `<decision_id>.json` file per ruling under
`data/data/rulings/`, deleting the downloaded `rulings.jsonl` once done.
Idempotent, runs once.

## Retrieval methods

The primary task is retrieving a case's `cited_rulings` from the ruling
corpus above. Five methods exist so far (see "Results" below for
measured numbers and discussion):

- **Constant** (`lib/metrics.py`'s `most_common_baseline`): the `k`
  most-frequently-cited rulings in the training split, predicted for
  every query regardless of input. The query-blind floor every real
  result has to clear.
- **BM25** (`lib/lexical/bm25.py`): lexical retrieval over each ruling's
  whole `considerations` text, with per-language (DE/FR/IT) tokenization
  and stemming via nltk's Snowball stemmer (`tokenize()`).
- **BM25, chunked** (`lib/lexical/bm25_chunked.py`): the same lexical
  scoring, but both sides are split into 512-token windows first (see
  `lib/chunking.py`) and a ruling scores as its best-matching chunk -
  chunks are roughly equal length, so this also avoids whole-document
  BM25's length-normalization penalty on long rulings.
- **idf-overlap** (`lib/lexical/idf_overlap.py`): a ruling scores as the
  summed corpus-wide rarity of the word stems it shares with the query,
  each stem counted once regardless of how often it recurs - BM25
  without term-frequency saturation or length normalization, i.e. "how
  far do rare shared words alone get you."
- **Dense** (`lib/dense/retrieval.py`, `lib/dense/embedding_model.py`):
  `intfloat/multilingual-e5-small` sentence embeddings, both sides
  chunked the same way as chunked BM25 and scored by best-matching
  chunk. Zero-shot by default; pass `--model-path` to score with a
  fine-tuned model instead (see "Fine-tuning" below).

The query is `lib/query.py`'s masked, concatenated `facts`+`considerations`
(citations are masked first, otherwise retrieval degenerates into citation lookup).
Every method caches its index under `data/indexes/` on first build
(idempotent - instant on every later run).

Retrieval quality is measured with `lib/metrics.py`'s `top_k_recall`,
`top_k_ndcg`, and `mrr` - all pure functions, unit-tested against
hand-computed values, not just example-based assertions.

Run the evaluation (downloads `train`/`validation` and builds whichever
indexes are needed if not already cached, then prints every method's
results):

```bash
python3 -m scripts.evaluate
```

See `python3 -m scripts.evaluate --help` for all options, including
`--methods` (evaluate a subset), `--train-cases`/`--val-queries`/
`--restrict-corpus` (a small, fast dev subset instead of the full split -
see "Results" below for why dev-subset and full-split numbers aren't
comparable), and `--model-path` (evaluate a fine-tuned model's dense row).

## Fine-tuning

`intfloat/multilingual-e5-small` can be contrastively fine-tuned on this
corpus (needs a GPU in practice - see "Setup" above):

```bash
python3 -m scripts.mine_pairs                       # mine training pairs once
python3 -m scripts.train --name my-run               # fine-tune, saves to data/models/my-run/
python3 -m scripts.evaluate --model-path data/models/my-run --methods dense
```

- **`scripts/mine_pairs.py`** (`lib/finetuning/pair_mining.py`): a
  training example is a (case, cited ruling) citation, but both sides are
  many 512-token chunks and nothing says which chunks the citation is
  "about" - the chunked BM25 index's best-scoring (query chunk, ruling
  chunk) pair stands in for that.
- **`scripts/train.py`** (`lib/finetuning/batch_sampling.py`): fine-tunes
  with `CachedMultipleNegativesRankingLoss` (in-batch negatives only,
  GradCache for a large effective batch on a small GPU). Batches are
  built so no in-batch "negative" actually shares a case or a cited
  ruling with the query - otherwise it would be a false negative, not a
  real one.

Both scripts take `--train-cases`/`--seed` for the same dev-subset workflow
as `scripts.evaluate` - see `--help` on each for the full option list.

## Failure analysis

Two tools for reading *why* a method got specific cases wrong, not just its
aggregate score (`failure_analysis/`):

```bash
python3 -m failure_analysis.inspect_failures --method bm25-chunked   # console
python3 -m failure_analysis.report --method bm25-chunked             # HTML
```

- **`failure_analysis.inspect_failures`**: prints the worst-recall
  validation cases for a method, with the query text and every missed
  gold ruling's own text, so you can read why a specific case failed
  (truncation, cross-lingual mismatch, a genuinely different legal area,
  ...) instead of just seeing a number.
- **`failure_analysis.report`**: the same idea as a self-contained,
  filterable HTML page (`data/reports/<method>_failures.html`) - a
  sortable table of cases; clicking a row opens a side-by-side view of
  the query and one of its rulings with the words they share highlighted
  by corpus-wide rarity. Both tools take the same method/subset options as
  `scripts.evaluate` - see `--help` on each.

## Results

Retrieval performance, with how each number was measured. Metrics are means
over queries: Recall@10/50/100, nDCG@10 and MRR, against the `cited_rulings` target
(the task: retrieve the cited rulings from a case's masked text).

**Provenance legend**

| tag | meaning |
|-----|---------|
| **verified** | Re-run from a clean `git archive` export of the committed code, so no uncommitted code is involved. |
| **earlier run** | Measured earlier in the project, on code that was not at a recorded commit or was uncommitted at the time. Not re-run against a commit. |

Environment for the verified runs: GTX 1060 6 GB, Python 3.12, torch 2.14.0+cu126,
sentence-transformers 5.6.1, bm25s 0.3.11, model `intfloat/multilingual-e5-small`.

### 1. Dev subset (verified)

A small, deterministic subset for fast iteration (seed 0): 5,000 sampled train cases
(for the constant baseline), 500 sampled validation queries, and a reduced corpus of
exactly the 4,684 rulings those cases cite. Because every ruling in this corpus is
cited by someone, retrieval is easier than on the full corpus: **compare rows within
this table only, never against §2 or §3.**

Run from a clean export of the committed code. Commands:
`python3 -m scripts.evaluate --train-cases 5000 --val-queries 500 --restrict-corpus`
for the first four rows, and the same with
`--methods dense --model-path data/models/dev-inbatch-6400-b` for the last.

| method | Recall@10 | Recall@50 | Recall@100 | nDCG@10 | MRR |
|--------|-----------|-----------|------------|---------|-----|
| constant (top-100 of the 5,000 train cases) | 0.076 | 0.212 | 0.296 | 0.078 | 0.180 |
| BM25, whole document | 0.145 | 0.268 | 0.332 | 0.166 | 0.330 |
| BM25, chunked | 0.172 | 0.328 | 0.389 | 0.184 | 0.341 |
| dense, zero-shot, chunked | 0.105 | 0.196 | 0.255 | 0.117 | 0.244 |
| dense, fine-tuned (`dev-inbatch-6400-b`) | 0.121 | 0.226 | 0.295 | 0.143 | 0.281 |

The first four rows were also verified on the code from before the fine-tuning code
was added, with identical numbers, so they do not depend on the fine-tuning
code.

**The fine-tuned model** (`scripts/train.py --name dev-inbatch-6400-b --train-cases 5000
--max-pairs 6400`): 6,400 pairs sampled from the 31,356 BM25-mined pairs of the 5,000
train cases (2,973 distinct cases), one pass of 47 steps in about 15 minutes, mean train
loss 2.781. `CachedMultipleNegativesRankingLoss` with in-batch negatives only (no hard
negatives), effective batch 128 (mini-batch 8), learning rate 2e-5, scale 100. Batches
never repeat a case or a ruling (`lib/finetuning/batch_sampling.py`). Positives are the BM25
best-matching (query chunk, ruling chunk) pair of each citation (`lib/finetuning/pair_mining.py`).

Reading:
- Chunking improves BM25 on every metric (Recall@10 0.145 to 0.172, MRR 0.330 to 0.341),
  and chunked BM25 is clearly ahead of zero-shot dense (Recall@10 0.172 vs 0.105).
- Fine-tuning improves zero-shot dense on every metric (Recall@10 +15%, nDCG@10 +22%,
  MRR +15%) and lifts Recall@100 to the constant baseline's level, but it stays well
  below chunked BM25 (Recall@10 0.121 vs 0.172, MRR 0.281 vs 0.341).
- Zero-shot dense falls below the constant baseline at Recall@50 and Recall@100.
- One seed and 500 queries: treat differences of a point or two as noise; no confidence
  intervals yet.

**Run-to-run variation of the fine-tuned model.** Two independent training runs with
identical settings (printed training losses identical to three decimals) scored:

| run | Recall@10 | Recall@50 | Recall@100 | nDCG@10 | MRR |
|-----|-----------|-----------|------------|---------|-----|
| `dev-inbatch-6400` | 0.121 | 0.228 | 0.294 | 0.143 | 0.281 |
| `dev-inbatch-6400-b` | 0.121 | 0.226 | 0.295 | 0.143 | 0.281 |

Differences of about 0.002 between runs are therefore not meaningful (most likely GPU
floating-point nondeterminism). Only `dev-inbatch-6400-b` is cited in the table above.

### 2. 500 validation queries against the full corpus (earlier run)

The same 500 sampled validation queries and 5,000 train cases as §1, but retrieving from
the full 21,171-ruling corpus (no `--restrict-corpus`). Run after the subset
options were introduced; the working-tree state was not recorded.

| method | Recall@10 | Recall@50 | Recall@100 | nDCG@10 | MRR |
|--------|-----------|-----------|------------|---------|-----|
| constant | 0.076 | 0.212 | 0.296 | 0.078 | 0.180 |
| BM25, whole document | 0.107 | 0.203 | 0.258 | 0.114 | 0.230 |
| dense, zero-shot, chunked | 0.072 | 0.140 | 0.176 | 0.077 | 0.166 |

Compared with §3 these land close to the full-split numbers (BM25 Recall@10 0.107 vs
0.120, dense 0.072 vs 0.077), so `--val-queries 500` alone is a reasonable stand-in for
the full evaluation, and about 2 minutes instead of about 45 when the indexes are cached.
Comparing §1 with this table shows how much the restricted corpus inflates scores (BM25
Recall@10 0.145 vs 0.107).

### 3. Full validation split (earlier run)

12,305 validation queries against the full 21,171-ruling corpus. Not re-run against a
commit; see the provenance column. About 90.7% of gold rulings resolve in the corpus,
so no method can exceed that recall (a data limit, not a method limit).

| method | Recall@10 | Recall@50 | Recall@100 | nDCG@10 | MRR | provenance |
|--------|-----------|-----------|------------|---------|-----|------------|
| constant | 0.071 | 0.194 | 0.264 | 0.076 | 0.176 | 2026-09-29, earlier run |
| BM25, whole document | 0.120 | 0.216 | 0.264 | 0.123 | 0.245 | 2026-09-29, earlier run |
| dense, zero-shot, unchunked (512-token truncation) | 0.032 | 0.072 | 0.100 | 0.031 | 0.069 | 2026-09-29, earlier run |
| BM25, chunked | 0.132 | 0.238 | 0.293 | 0.129 | 0.251 | 2026-10-02, uncommitted working tree |
| dense, zero-shot, chunked | 0.077 | 0.150 | 0.191 | 0.077 | 0.162 | 2026-10-02, uncommitted working tree |

Chunking more than doubles dense retrieval (Recall@10 0.032 to 0.077) and lifts BM25
(0.120 to 0.132), but zero-shot dense still trails both BM25 variants and is level with
the constant baseline at Recall@10.

### 4. Caveats

- The verified run reuses index caches built earlier from this code (dense and
  whole-document BM25 for the dev corpus, and the chunked BM25 dev index built the same
  day). The chunking is unchanged, but the caches were not rebuilt per commit.
- The dev subset is a single seed and a small, easier corpus; §1, §2 and §3 are not
  comparable to each other.
- The fine-tuned model is a saved artifact under `data/models/` (not in git). It was
  trained with the command above from the working tree, with the fine-tuning files that
  were committed afterwards, not re-trained from the clean export. The
  evaluation itself did run from the clean export.
- No confidence intervals are reported yet.

### 5. Not yet evaluated

- A full-split run from a clean export (about 45 minutes), to turn §3 into verified
  numbers.
- Hybrid BM25 + dense fusion.
- Larger fine-tuning runs (all 31,356 pairs, more epochs), dropping weakly matched pairs
  (`--min-bm25-score`), and BM25 hard negatives. The fine-tuned model has not been
  evaluated on the full split or with the full corpus.

## Citing

If you use this code, cite it via `CITATION.cff` (GitHub's "Cite this repository"
button), and cite the dataset paper listed under "License" below.

## License

Code is GPL-3.0-or-later (see `LICENSE`).

The data comes from two Hugging Face datasets curated by Stern et al.
([SCALE, arXiv:2306.09237](https://arxiv.org/abs/2306.09237)):
[`rcds/swiss_doc2doc_ir`](https://huggingface.co/datasets/rcds/swiss_doc2doc_ir) and
[`rcds/swiss_leading_decisions`](https://huggingface.co/datasets/rcds/swiss_leading_decisions).
Both dataset cards disagree with themselves on their own license: the page's
metadata tag says CC-BY-SA-4.0 (Attribution-ShareAlike), but the card's own
"Licensing Information" text says plain CC-BY-4.0 (no ShareAlike) - identical
wording on both cards, so it's a deliberate statement, not a one-off typo,
but still unresolved as of 2026-10-09. This project assumes the stricter
**CC-BY-SA-4.0** applies.

In practice, under that assumption:
- **Attribution**: the datasets are cited above.
- **ShareAlike**: `data/` (the raw corpus, mined training pairs, generated
  failure reports, fine-tuned model checkpoints) is gitignored and never
  committed, so this repository does not itself redistribute the dataset or
  anything derived from it. If that ever changes - publishing mined pairs,
  a processed corpus subset, or the fine-tuned model - that artifact would
  need to carry CC-BY-SA-4.0 (not this repo's GPL license) plus attribution,
  not a more permissive license.
- The fine-tuned model's weights are a separate, still-open question:
  whether trained weights count as a "derivative work" of CC-BY-SA text is
  unsettled even among lawyers. Not yet addressed here, since the model
  stays local for now - treat publishing it as its own decision, not
  something this license section already clears.
