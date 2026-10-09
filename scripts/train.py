"""
Contrastively fine-tune the embedding model on mined (query chunk, ruling chunk) pairs
with in-batch negatives only.

`CachedMultipleNegativesRankingLoss` (GradCache) gives a large effective batch - and so
many in-batch negatives - on a 6 GB GPU by embedding it in small mini-batches, at about
twice the compute per step. Batches never repeat a case or a ruling, so no in-batch
"negative" is actually relevant (see `lib.finetuning.batch_sampling`).
"""

import cyclopts
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

from lib.atomic_dir import atomic_dir_write
from lib.dense.embedding_model import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    load_model_for_training,
)
from lib.finetuning.batch_sampling import make_distinct_sampler
from lib.finetuning.pair_mining import load_pairs, pairs_name
from lib.sampling import sample_items
from paths import MODELS_DIR


def train(
    *,
    name: str,
    train_cases: int | None = None,
    batch_size: int = 128,
    mini_batch_size: int = 8,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    scale: float = 100.0,
    min_bm25_score: float = 0.0,
    max_pairs: int | None = None,
    seed: int = 0,
) -> None:
    """
    Fine-tune and save the model under data/models/<name>/.

    :param name: The run's name; the model directory must not exist yet (a dense index
        cache is keyed on it, so reusing a name would load a stale index).
    :param train_cases: Train on the pairs mined from this many sampled train cases (see
        `scripts.mine_pairs`, which must have run with the same `train_cases` and
        `seed`); on the full train split's pairs if omitted.
    :param batch_size: The effective batch size, i.e. the in-batch negatives per query
        plus one.
    :param mini_batch_size: How many pairs the GPU embeds at once; lower it if the GPU
        runs out of memory. Doesn't change the result, only speed and memory. 16 already
        runs out of memory on the 6 GB GTX 1060 at full 512-token length.
    :param epochs: How many passes over the pairs.
    :param learning_rate: The peak learning rate (10% linear warmup, then linear decay).
    :param scale: The similarity scale (1 / temperature) of the contrastive loss; e5 was
        pretrained with temperature 0.01.
    :param min_bm25_score: Drop mined pairs whose BM25 score is below this, as they are
        the least trustworthy; the default keeps every pair.
    :param max_pairs: Train on a random sample of this many pairs, for a quicker run;
        all of them if omitted.
    :param seed: The seed of the train-case sample the pairs were mined from; also the
        shuffling and sampling seed.
    """
    output_dir = MODELS_DIR / name
    if output_dir.is_dir():
        raise FileExistsError(f"{output_dir} already exists - pick a new --name")

    pairs = [
        p
        for p in load_pairs(name=pairs_name(train_cases=train_cases, seed=seed))
        if p.bm25_score >= min_bm25_score
    ]
    if max_pairs is not None:
        pairs = sample_items(items=pairs, n=max_pairs, seed=seed)
    print(
        f"training on {len(pairs)} pairs from {len({p.case_id for p in pairs})} cases"
    )

    dataset = Dataset.from_dict(
        {
            "anchor": [f"{QUERY_PREFIX}{p.query_chunk}" for p in pairs],
            "positive": [f"{PASSAGE_PREFIX}{p.ruling_chunk}" for p in pairs],
        }
    )
    embedding_model = load_model_for_training()
    with atomic_dir_write(final_dir=output_dir) as tmp_dir:
        tmp_dir.mkdir(parents=True)
        args = SentenceTransformerTrainingArguments(
            output_dir=str(tmp_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            # Pascal GPUs (GTX 1060) have no fast half precision
            fp16=False,
            bf16=False,
            batch_sampler=make_distinct_sampler(
                case_ids=[p.case_id for p in pairs],
                ruling_ids=[p.ruling_id for p in pairs],
            ),
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            seed=seed,
        )
        trainer = SentenceTransformerTrainer(
            model=embedding_model,
            args=args,
            train_dataset=dataset,
            loss=CachedMultipleNegativesRankingLoss(
                embedding_model,
                scale=scale,
                mini_batch_size=mini_batch_size,
                show_progress_bar=False,
            ),
        )
        trainer.train()
        embedding_model.save(str(tmp_dir))
    print(f"saved {output_dir}")


if __name__ == "__main__":
    cyclopts.run(train)
