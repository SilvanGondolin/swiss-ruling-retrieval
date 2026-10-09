# swiss-ruling-retrieval

Multilingual (DE/FR/IT) Swiss legal retrieval: given a Federal Supreme Court
case, retrieve the rulings it cites. Baselines (constant, BM25, chunked BM25,
IDF overlap, zero-shot dense), contrastive fine-tuning of `multilingual-e5-small`, and
a hybrid of chunked BM25 and the fine-tuned model.

## 1. Install the environment

Python >= 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # CPU, or a recent GPU
# old Pascal-era GPU (e.g. GTX 1060): pip install -r requirements-cu126.txt
python3 -m pytest                          # optional: run the tests
```

## 2. Get the data

Public on the Hugging Face hub, no account or token needed:
[`rcds/swiss_doc2doc_ir`](https://huggingface.co/datasets/rcds/swiss_doc2doc_ir)
(cases) and
[`rcds/swiss_leading_decisions`](https://huggingface.co/datasets/rcds/swiss_leading_decisions)
(the 21,171-ruling corpus). Everything is downloaded into `data/` the first time
it is needed, so any command below fetches what it requires. To fetch and check
a split on its own:

```bash
python3 -m scripts.inspect_data --split=train        # or: validation, test
python3 -m scripts.inspect_data --split=validation
```

Preparing a split (once, when it is first downloaded) removes every cited ruling that
is not in the 21,171-ruling corpus (about 9% of the raw citations), and then the cases
left without any cited ruling, so every remaining citation can be retrieved. A copy
downloaded by an earlier version is prepared in place the first time it is loaded.

## 3. Reproduce the results

Metrics are Recall@10/50/100, nDCG@10 and MRR on the validation split.

Full validation split against the full corpus: `python3 -m scripts.evaluate` with no
arguments. Preparing the splits removed the cited rulings that are not in the corpus
(10,382 citations) and 89 cases left without any cited ruling, so 12,216 validation
queries remain (the cases with at least one cited ruling), evaluated against 21,171
rulings. A single run, no confidence intervals.

| method | Recall@10 | Recall@50 | Recall@100 | nDCG@10 | MRR |
|---|---|---|---|---|---|
| constant (most-cited rulings) | 0.071 | 0.203 | 0.294 | 0.075 | 0.178 |
| BM25, whole document | 0.132 | 0.236 | 0.288 | 0.130 | 0.247 |
| BM25, chunked | 0.144 | 0.259 | 0.317 | 0.135 | 0.252 |
| IDF overlap | 0.062 | 0.124 | 0.169 | 0.057 | 0.115 |
| dense, zero-shot (chunked) | 0.084 | 0.165 | 0.210 | 0.081 | 0.163 |
| dense, fine-tuned, 1 epoch (chunked) | 0.110 | 0.218 | 0.276 | 0.104 | 0.197 |
| dense, fine-tuned, 2 epochs (chunked) | 0.116 | 0.224 | 0.286 | 0.106 | 0.195 |
| dense, fine-tuned, 6 epochs (chunked) | 0.143 | 0.259 | 0.325 | 0.129 | 0.217 |
| dense, fine-tuned, 6 epochs + 1 on new pairs (chunked) | 0.157 | 0.280 | 0.349 | 0.144 | 0.237 |
| hybrid: chunked BM25 + that fine-tuned dense model | 0.186 | 0.328 | 0.399 | 0.173 | 0.299 |

- Chunked BM25 is ahead of whole-document BM25 on every metric.
- Whole-document BM25 falls just below the constant baseline at Recall@100 (0.288 vs
  0.294); chunked BM25 stays above it (0.317).
- IDF overlap is below the constant baseline on every metric.
- Zero-shot dense is above the constant baseline at Recall@10 and nDCG@10, and below
  it at Recall@50, Recall@100 and MRR.
- Fine-tuning improves the dense model on every metric except MRR at 2 epochs: one
  epoch takes Recall@10 from 0.084 to 0.110 and Recall@100 from 0.210 to 0.276, two
  epochs reach 0.116 and 0.286 (MRR 0.197 to 0.195), and six epochs 0.143 and 0.325.
- Six epochs on the same 60,000 pairs reach Recall@100 0.325, above chunked BM25
  (0.317) and the constant baseline (0.294). One more epoch on 60,000 different pairs
  takes it to 0.349 and beats chunked BM25 on Recall@10 (0.157 vs 0.144), Recall@50
  (0.280 vs 0.259) and nDCG@10 (0.144 vs 0.135). Chunked BM25 stays ahead on MRR
  (0.252 vs 0.237).
- The hybrid is the best method on every metric. It fuses the chunked BM25 and the
  fine-tuned dense ranking by reciprocal rank fusion (the top 200 of each, score
  1 / (60 + rank) summed over the two, no tuning). Against its better parent it adds
  0.029 Recall@10 (0.186 vs 0.157), 0.050 Recall@100 (0.399 vs 0.349) and, because
  BM25 ranks the first hits well, 0.047 MRR (0.299 vs 0.252).

Each fine-tuning epoch is one pass over 60,000 randomly sampled training pairs (35,545
cases), about 2.5 hours on a GTX 1060 6 GB. The training pairs are mined on the first
run: for each citation, chunked BM25 picks the query chunk and the chunk of the cited
ruling with the highest BM25 score, as the citation does not say which chunks it is
about. It needs a GPU:

```bash
python3 -m scripts.train --name ft --max-pairs 60000 --epochs 1
python3 -m scripts.evaluate --methods dense --model-path data/models/ft
# second epoch, on top of the first (same pairs, new optimizer and schedule)
python3 -m scripts.train --name ft-2ep --init-model ft --max-pairs 60000 --epochs 1
python3 -m scripts.evaluate --methods dense --model-path data/models/ft-2ep
# four more epochs in one run (about 10 hours), reshuffled each epoch
python3 -m scripts.train --name ft-6ep --init-model ft-2ep --max-pairs 60000 --epochs 4
python3 -m scripts.evaluate --methods dense --model-path data/models/ft-6ep
# one epoch on a different sample of 60,000 pairs (about 88% of them new)
python3 -m scripts.train --name ft-new60k --init-model ft-6ep --max-pairs 60000 --sample-seed 1 --epochs 1
python3 -m scripts.evaluate --methods dense --model-path data/models/ft-new60k
# the hybrid row: chunked BM25 and the fine-tuned model, fused (about 40 minutes)
python3 -m scripts.evaluate --methods hybrid --model-path data/models/ft-new60k
```

Training writes a checkpoint every 100 steps to `data/models/ft.checkpoints/`. If the
run is interrupted, rerun the same command and it continues from the latest checkpoint
(`--no-resume` starts over). To train further on a finished model, start a new run
from it with `--init-model`, as above: its weights are kept, the optimizer and
learning-rate schedule start anew.

## Why the scores are low

All methods leave a lot of room for improvement. Three reasons, in decreasing order of
how well we can back them with numbers (`python3 -m scripts.task_statistics` prints
the statistics below for the prepared splits):

**1. The task is a needle in a haystack.** A validation query has 8.2 relevant rulings
on average (median 6, up to 111) among 21,171 candidates. That is 0.04% of the corpus:
about 2,600 distractors for every relevant ruling, and a random ranking gets a
Recall@10 of about 0.0005. The queries are also long, whole decisions whose citations
sit in single sentences, so most of the text says nothing about any one cited ruling.
Even a perfect ranking cannot reach a Recall@10 of 1: with this many relevant rulings
per query, the best possible Recall@10 averages 0.894 (Recall@50 and @100 can reach
about 1.0).

**2. This is extreme multi-class retrieval, with a long tail.** Every ruling is a
class, and the classes are very uneven:
- 56.7% of the rulings are never cited in the train split, so there is nothing to learn
  from for them; 9.3% of the validation citations point to such a ruling, and 15.1%
  point to a ruling cited fewer than 10 times in train.
- The 100 most cited rulings make up 32.1% of the train citations (the top 1,000 make
  up 73.7%). That is why the constant "most cited" baseline is competitive at
  Recall@100, and why methods have to beat popularity, not only chance.
- 38.8% of the validation citations point to a ruling in another language than the
  citing case. Lexical methods (BM25, IDF overlap) cannot find those at all.

**3. The learned model is small and trained on a limited set of pairs.** The dense retriever
is `multilingual-e5-small`, chosen because the experiments ran on a single 6 GB GPU. It
embeds at most 512 tokens at a time, so every ruling and query is split into chunks
and scored by its best chunk. Fine-tuning used in-batch negatives, an effective batch
of 128 and two samples of 60,000 of the 498,724 mined pairs, for seven epochs in all.
Recall@100 went from 0.286 after two epochs to 0.325 after six on the first sample, and
to 0.349 after one epoch on the second. We did not use all the pairs or a larger model,
so how much of the remaining gap they would close is untested.

## License and citing

Code: GPL-3.0-or-later (`LICENSE`). Cite the code via `CITATION.cff` and the data
via the SCALE paper
([Rasiah et al., arXiv:2306.09237](https://arxiv.org/abs/2306.09237)). The two
dataset cards state conflicting licenses (CC-BY-SA-4.0 in the metadata, CC-BY-4.0
in the text); this project assumes the stricter CC-BY-SA-4.0. `data/` is
gitignored, so this repository redistributes no dataset content.
