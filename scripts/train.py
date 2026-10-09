"""
Contrastively fine-tune the embedding model on mined (query chunk, ruling chunk) pairs
with in-batch negatives only.

`CachedMultipleNegativesRankingLoss` (GradCache) gives a large effective batch - and so
many in-batch negatives - on a 6 GB GPU by embedding it in small mini-batches, at about
twice the compute per step. Batches never repeat a case or a ruling, so no in-batch
"negative" is actually relevant (see `lib.finetuning.batch_sampling`).
"""

import json
import shutil
from pathlib import Path

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
from lib.finetuning.batch_sampling import DistinctSamplerFactory
from lib.finetuning.pair_mining import load_pairs, pairs_name
from lib.sampling import sample_items
from paths import MODELS_DIR


_SETTINGS_FILENAME = "run_settings.json"
# The trainer writes this last into a checkpoint, so its presence marks it complete.
_CHECKPOINT_DONE_FILENAME = "trainer_state.json"


def _latest_complete_checkpoint(*, checkpoint_dir: Path) -> Path | None:
    """
    :param checkpoint_dir: Where the run's checkpoints are written.
    :returns: The newest checkpoint that was written completely - a run killed while
        saving leaves a partial newest one - or None if there is none.
    """
    by_step = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    return next((c for c in by_step if (c / _CHECKPOINT_DONE_FILENAME).is_file()), None)


def _checkpoint_to_resume(
    *, checkpoint_dir: Path, resume: bool, settings: dict[str, object]
) -> str | None:
    """
    Decide whether this run continues an interrupted one, and record its settings.

    :param checkpoint_dir: Where the run's checkpoints are written.
    :param resume: Whether to continue from an existing checkpoint.
    :param settings: The arguments that shape the run, to compare with the interrupted
        run's: resuming under different ones would silently mix two runs.
    :returns: The latest checkpoint's path to continue from, or None to start fresh.
    :raises ValueError: If there is a checkpoint from a run with different settings.
    """
    settings_path = checkpoint_dir / _SETTINGS_FILENAME
    last = _latest_complete_checkpoint(checkpoint_dir=checkpoint_dir)
    if resume and last is not None:
        saved = json.loads(settings_path.read_text())
        if saved != settings:
            raise ValueError(
                f"{checkpoint_dir} is from a run with other settings {saved}; "
                "rerun with the same arguments, or with --no-resume to start over"
            )
        print(f"resuming from {last}")
        return str(last)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    checkpoint_dir.mkdir(parents=True)
    settings_path.write_text(json.dumps(settings))
    return None


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
    sample_seed: int | None = None,
    init_model: str | None = None,
    resume: bool = True,
    save_steps: int = 100,
) -> None:
    """
    Fine-tune and save the model under data/models/<name>/.

    A checkpoint (weights, optimizer, schedule position) is written to
    data/models/<name>.checkpoints/ every `save_steps` steps. Running the same command
    again after an interruption continues from the latest checkpoint; the directory is
    deleted once the run finishes.

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
    :param sample_seed: The seed of the `max_pairs` sample; `seed` if omitted. Another
        value draws a different sample from the same mined pairs (about
        `max_pairs` / all pairs of it overlaps the earlier one), e.g. to train on
        new pairs on top of `init_model`.
    :param init_model: The name of a finished run (under data/models) to start from
        instead of the pretrained model, for more epochs on top of it. Only its weights
        are kept: the optimizer and learning-rate schedule start anew, and with the
        same seed the pairs come in the same order as in that run's first epoch.
    :param resume: Continue from the latest checkpoint if there is one; False deletes
        it and starts over. Resuming needs the same arguments as the interrupted run.
    :param save_steps: How many steps between checkpoints.
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
        pairs = sample_items(
            items=pairs, n=max_pairs, seed=seed if sample_seed is None else sample_seed
        )
    print(
        f"training on {len(pairs)} pairs from {len({p.case_id for p in pairs})} cases"
    )

    dataset = Dataset.from_dict(
        {
            "anchor": [f"{QUERY_PREFIX}{p.query_chunk}" for p in pairs],
            "positive": [f"{PASSAGE_PREFIX}{p.ruling_chunk}" for p in pairs],
        }
    )
    embedding_model = load_model_for_training(
        model_path=None if init_model is None else MODELS_DIR / init_model
    )
    # Outside the atomic directory, so it survives a run that is killed before the end.
    checkpoint_dir = MODELS_DIR / f"{name}.checkpoints"
    resume_from = _checkpoint_to_resume(
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        settings={
            "train_cases": train_cases,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "scale": scale,
            "min_bm25_score": min_bm25_score,
            "max_pairs": max_pairs,
            "seed": seed,
            "sample_seed": sample_seed,
            "init_model": init_model,
        },
    )
    with atomic_dir_write(final_dir=output_dir) as tmp_dir:
        tmp_dir.mkdir(parents=True)
        args = SentenceTransformerTrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            # Pascal GPUs (GTX 1060) have no fast half precision
            fp16=False,
            bf16=False,
            batch_sampler=DistinctSamplerFactory(
                case_ids=[p.case_id for p in pairs],
                ruling_ids=[p.ruling_id for p in pairs],
            ),
            logging_steps=5,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
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
        trainer.train(resume_from_checkpoint=resume_from)
        embedding_model.save(str(tmp_dir))
    shutil.rmtree(checkpoint_dir)
    print(f"saved {output_dir}")


if __name__ == "__main__":
    cyclopts.run(train)
