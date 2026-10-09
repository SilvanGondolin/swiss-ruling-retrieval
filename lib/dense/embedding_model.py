"""The shared dense embedding model, its input conventions, and its chunking."""

from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lib.chunking import TokenSpan, chunk_text

MODEL_NAME = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


@lru_cache(maxsize=None)
def model(*, path: Path | None = None) -> SentenceTransformer:
    """
    :param path: A fine-tuned model's directory; the pretrained `MODEL_NAME` if
        omitted.
    :returns: The cached, shared embedding model instance.
    """
    return SentenceTransformer(MODEL_NAME if path is None else str(path))


@lru_cache(maxsize=None)
def tokenizer() -> PreTrainedTokenizerBase:
    """:returns: The embedding model's tokenizer, loaded without the model itself."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model_for_training() -> SentenceTransformer:
    """
    Load a fresh, uncached instance of the pretrained `MODEL_NAME`.

    Training updates a model's weights in place - returning the `model()` singleton
    here would permanently contaminate that shared cache entry with the fine-tuned
    weights for the rest of the process.

    :returns: A new `SentenceTransformer`, independent of the `model()` cache.
    """
    return SentenceTransformer(MODEL_NAME)


def chunk_for_model(*, text: str, prefix: str) -> list[str]:
    """
    Split `text` into windows that, with `prefix` and the model's special tokens added,
    fit the model's max_seq_length.

    :param text: The text to split.
    :param prefix: The model's input prefix, which the caller prepends to every chunk
        before encoding - only its length is used here, to size the windows.
    :returns: The chunks, without `prefix`.
    """
    tok = tokenizer()
    # How many tokens the prefix ("query: " / "passage: ") takes up once tokenized, so
    # that much of the model's window can be reserved for it. Special tokens are left
    # out here because they are counted separately below.
    prefix_length = len(tok(prefix, add_special_tokens=False).input_ids)
    # One token of margin: a chunk that starts mid-word gets an extra "\u2581" token
    # when re-tokenized as standalone text.
    window_size = (
        tok.model_max_length - tok.num_special_tokens_to_add() - prefix_length - 1
    )
    # The (start, end) character span of each token in `text`, so `chunk_text` can cut
    # at token boundaries and slice the original string. Special tokens are left out
    # since they have no position in the text, and `verbose=False` silences the
    # "sequence longer than the maximum length" warning - we split long texts on
    # purpose, so it doesn't apply.
    token_offsets = [
        TokenSpan(start=start, end=end)
        for start, end in tok(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        ).offset_mapping
    ]
    return chunk_text(text=text, token_offsets=token_offsets, window_size=window_size)
