import math

import bm25s

from lib.language import Language
from lib.lexical.bm25 import (
    DocumentFrequencies,
    document_frequencies,
    stem_word,
    tokenize,
)


def test_lowercases():
    assert tokenize("BESCHWERDE", Language.DE) == tokenize("beschwerde", Language.DE)


def test_removes_german_stopwords():
    tokens = tokenize("der Kläger und die Beklagte", Language.DE)
    assert "der" not in tokens
    assert "und" not in tokens
    assert "die" not in tokens


def test_removes_french_stopwords():
    tokens = tokenize("le recourant et la partie", Language.FR)
    assert "le" not in tokens
    assert "et" not in tokens
    assert "la" not in tokens


def test_removes_italian_stopwords():
    tokens = tokenize("il ricorrente e la parte", Language.IT)
    assert "il" not in tokens
    assert "e" not in tokens
    assert "la" not in tokens


def test_stems_german_inflections_to_a_shared_root():
    # singular/plural inflections of "Kläger" (plaintiff) share a stem - but note
    # Snowball doesn't decompose German compounds, so e.g. "Beschwerdeführer"
    # stems separately from "Beschwerde", not to a shared root
    singular = tokenize("der Kläger", Language.DE)
    plural = tokenize("den Klägern", Language.DE)
    assert set(singular) & set(plural)


def test_strips_punctuation_and_short_tokens():
    tokens = tokenize("Art. 95 BGG; siehe oben.", Language.DE)
    assert "." not in tokens
    assert ";" not in tokens
    # single-char tokens like the digit-less remainder of "95" are still kept
    # if >= 2 chars; this just checks punctuation itself isn't a token
    assert all(len(t) >= 2 for t in tokens)


def test_empty_string():
    assert tokenize("", Language.DE) == []


def test_stem_word_lowercases_and_stems_like_tokenize():
    assert stem_word("Verfahren", Language.DE) == tokenize("verfahren", Language.DE)[0]


def test_stem_word_of_a_stopword_is_none():
    assert stem_word("Der", Language.DE) is None


def _index(*, documents: list[list[str]]) -> bm25s.BM25:
    index = bm25s.BM25()
    index.index(documents, show_progress=False)
    return index


def test_document_frequencies_count_rulings_not_occurrences():
    frequencies = document_frequencies(
        index=_index(documents=[["a", "b", "a"], ["b", "c"], ["b"], ["d", "b", "b"]]),
        document_languages=[Language.DE] * 4,
    )
    assert frequencies.counts == {"a": 1, "b": 4, "c": 1, "d": 1}
    assert frequencies.totals == {Language.DE: 4}


def test_document_frequencies_total_rulings_per_language():
    frequencies = document_frequencies(
        index=_index(documents=[["a"], ["a"], ["b"]]),
        document_languages=[Language.DE, Language.FR, Language.FR],
    )
    assert frequencies.totals == {Language.DE: 1, Language.FR: 2}


def test_idf_is_higher_for_rarer_stems():
    frequencies = DocumentFrequencies(
        counts={"common": 90, "rare": 2}, totals={Language.DE: 100}
    )
    common = frequencies.idf(stem="common", language=Language.DE)
    rare = frequencies.idf(stem="rare", language=Language.DE)
    unseen = frequencies.idf(stem="unseen", language=Language.DE)
    assert 0 < common < rare < unseen


def test_idf_matches_the_bm25_formula():
    frequencies = DocumentFrequencies(counts={"s": 9}, totals={Language.DE: 100})
    assert frequencies.idf(stem="s", language=Language.DE) == math.log(
        1 + (100 - 9 + 0.5) / (9 + 0.5)
    )


def test_idf_clamps_a_count_above_the_language_total():
    # a stem shared by two languages can be counted in more rulings than one has
    frequencies = DocumentFrequencies(counts={"s": 500}, totals={Language.DE: 100})
    assert frequencies.idf(stem="s", language=Language.DE) > 0
