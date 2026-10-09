"""Splitting long texts into fixed-size token windows for dense embedding."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenSpan:
    """
    :ivar start: The character offset where the token starts in its text.
    :ivar end: The character offset just past the token's last character.
    """

    start: int
    end: int


def chunk_text(
    *, text: str, token_offsets: list[TokenSpan], window_size: int
) -> list[str]:
    """
    Split `text` into consecutive, non-overlapping chunks of `window_size` tokens each
    (the last chunk may be shorter). Chunks are slices of the original text, so
    concatenating them loses only the whitespace between chunk boundaries.

    :param text: The text to split.
    :param token_offsets: The character span of each of `text`'s tokens, in order - from
        a fast tokenizer with `return_offsets_mapping=True`, without special
        tokens.
    :param window_size: The maximum number of tokens per chunk.
    :returns: At least one chunk: a text with no tokens comes back unchanged as a
        single chunk, so every document stays represented.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if not token_offsets:
        return [text]
    return [
        text[window[0].start : window[-1].end]
        for window in (
            token_offsets[i : i + window_size]
            for i in range(0, len(token_offsets), window_size)
        )
    ]


def chunk_starts(*, chunk_counts: list[int]) -> np.ndarray:
    """
    :param chunk_counts: How many chunks each document has, in document order. Each
        must be at least 1.
    :returns: The row of each document's first chunk, when all documents' chunks are
        stacked in order - e.g. `[2, 1, 3]` gives `[0, 2, 3]`.
    """
    if any(count < 1 for count in chunk_counts):
        raise ValueError("every document needs at least one chunk")
    starts = np.zeros(len(chunk_counts), dtype=np.int64)
    np.cumsum(chunk_counts[:-1], out=starts[1:])
    return starts
