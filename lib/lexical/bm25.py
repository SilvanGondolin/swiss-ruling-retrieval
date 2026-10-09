"""
BM25 retrieval baseline over the rcds/swiss_leading_decisions ruling corpus.

Uses a single combined index over all languages, but tokenizes/stems each document and
query with its own language's Snowball stemmer and stopword list (language-appropriate
tokenization/stemming). Cross- lingual matches (a German query lexically matching a
French document) are essentially impossible under this scheme, which is fine: BM25 is
lexical, so a single shared index doesn't lose anything a per-language index would give
us, and it's simpler.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import bm25s
import nltk
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from lib.atomic_dir import atomic_dir_write
from lib.decision_ids import read_decision_ids, write_decision_ids
from lib.index_cache import cache_name
from lib.language import Language
from lib.rulings import iter_rulings
from paths import DATA_DIR, INDEX_DIR

_TOKEN_PATTERN = re.compile(r"\b\w\w+\b")
_CACHE_BASE = "bm25_index"
_NLTK_DATA_DIR = DATA_DIR / "nltk_data"


def _ensure_stopwords_present() -> None:
    """
    Download nltk's `stopwords` data package into data/nltk_data/ if necessary, and put
    that directory first on nltk's search path.

    The package isn't installed with `nltk` itself, so a fresh machine doesn't have it.
    Idempotent: only downloads if the directory doesn't exist yet; written atomically
    (see `lib.atomic_dir`).
    """
    if not _NLTK_DATA_DIR.is_dir():
        with atomic_dir_write(final_dir=_NLTK_DATA_DIR) as tmp_dir:
            tmp_dir.mkdir(parents=True)
            nltk.download(
                "stopwords", download_dir=str(tmp_dir), quiet=True, raise_on_error=True
            )
    if str(_NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(_NLTK_DATA_DIR))


@lru_cache(maxsize=None)
def _stemmer(language: Language) -> SnowballStemmer:
    """
    :param language: The stemmer's language.
    :returns: A cached `SnowballStemmer` for that language.
    """
    return SnowballStemmer(language.nltk_name)


@lru_cache(maxsize=None)
def _stopwords(language: Language) -> frozenset[str]:
    """
    :param language: The stopword list's language.
    :returns: A cached, frozen set of that language's stopwords.
    """
    _ensure_stopwords_present()
    return frozenset(nltk.corpus.stopwords.words(language.nltk_name))


@lru_cache(maxsize=None)
def stem_word(word: str, language: Language) -> str | None:
    """
    Reduce one word the way `tokenize` does: lowercase, then stem.

    :param word: A single word.
    :param language: The word's language.
    :returns: The word's stem, or None if it is a stopword.
    """
    lowered = word.lower()
    if lowered in _stopwords(language):
        return None
    return _stemmer(language).stem(lowered)


def tokenize(text: str, language: Language) -> list[str]:
    """
    Lowercase, tokenize, remove stopwords, and stem `text` using `language`'s Snowball
    stemmer and stopword list.

    :param text: The text to tokenize.
    :param language: The text's language.
    :returns: The stemmed, stopword-filtered tokens.
    """
    stemmer = _stemmer(language)
    stop = _stopwords(language)
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return [stemmer.stem(t) for t in tokens if t not in stop]


@dataclass(frozen=True)
class RulingIndex:
    index: bm25s.BM25
    decision_ids: list[str]


def build_ruling_index(*, ruling_ids: frozenset[str] | None = None) -> RulingIndex:
    """
    Build a BM25 index over the whole ruling corpus's `considerations` text - the
    substantive legal reasoning a citing case would actually be drawing on.
    Downloads/expands the corpus first if necessary (see `lib.rulings`). Cached to
    disk under data/indexes/ (one directory per distinct set of rulings, see
    `lib.index_cache`) - idempotent, only built once (building it
    takes a while: mostly per-language tokenization/stemming of 21K+ documents).

    :param ruling_ids: Only index these rulings (a reduced corpus, for fast
        iteration); all of them if omitted.
    :returns: The index, with `decision_ids` aligned to the indexed documents.
    """
    index_dir = INDEX_DIR / cache_name(base=_CACHE_BASE, ruling_ids=ruling_ids)
    if index_dir.is_dir():
        index = bm25s.BM25.load(str(index_dir), load_corpus=False, show_progress=False)
        return RulingIndex(
            index=index, decision_ids=read_decision_ids(directory=index_dir)
        )

    decision_ids: list[str] = []
    corpus_tokens: list[list[str]] = []
    for ruling in tqdm(
        iter_rulings(ruling_ids=ruling_ids), desc="tokenizing rulings for BM25"
    ):
        decision_ids.append(ruling.decision_id)
        corpus_tokens.append(tokenize(ruling.considerations, ruling.language))

    index = bm25s.BM25()
    index.index(corpus_tokens, show_progress=False)

    with atomic_dir_write(final_dir=index_dir) as tmp_dir:
        index.save(str(tmp_dir), show_progress=False)
        write_decision_ids(directory=tmp_dir, decision_ids=decision_ids)

    return RulingIndex(index=index, decision_ids=decision_ids)


def retrieve(
    ruling_index: RulingIndex, query: str, language: Language, k: int
) -> list[str]:
    """
    Retrieve the top-k ruling decision_ids for `query`.

    :param ruling_index: A ruling index built by `build_ruling_index`.
    :param query: The query text - already citation-masked, see `lib.query`.
    :param language: The query's language.
    :param k: How many results to return.
    :returns: Up to `k` decision_ids, best first.
    """
    query_tokens = tokenize(query, language)
    results, _ = ruling_index.index.retrieve(
        query_tokens=[query_tokens],
        corpus=ruling_index.decision_ids,
        k=k,
        show_progress=False,
    )
    if len(results) != 1:
        raise RuntimeError("Expected exactly 1 result, got {}".format(len(results)))
    return list(results[0])


@dataclass(frozen=True)
class DocumentFrequencies:
    """
    How many rulings of each language contain each stem - the basis of a word's rarity.

    :ivar counts: The number of rulings containing each stem.
    :ivar totals: The number of rulings in each language.
    """

    counts: dict[str, int]
    totals: dict[Language, int]

    def idf(self, *, stem: str, language: Language) -> float:
        """
        :param stem: A stem, as `stem_word` returns it.
        :param language: The language of the text the stem occurs in.
        :returns: BM25's inverse document frequency of `stem` among `language`'s
            rulings: about 0 for a stem in nearly every ruling, growing the rarer it
            is, up to ~ln(2 * number of rulings) for a stem in none.
        """
        total = self.totals[language]
        count = min(self.counts.get(stem, 0), total)
        return math.log(1 + (total - count + 0.5) / (count + 0.5))


def document_frequencies(
    *, index: bm25s.BM25, document_languages: list[Language]
) -> DocumentFrequencies:
    """
    :param index: A BM25 index of rulings, see `build_ruling_index`.
    :param document_languages: The language of each indexed ruling, in index order.
    :returns: How many rulings contain each of the index's stems. Read off the index's
        per-stem score columns, which hold one entry per ruling containing the stem.
    """
    column_starts = index.scores["indptr"]
    counts = {
        stem: int(column_starts[stem_id + 1] - column_starts[stem_id])
        for stem, stem_id in index.vocab_dict.items()
        if stem and stem_id + 1 < len(column_starts)
    }
    return DocumentFrequencies(counts=counts, totals=dict(Counter(document_languages)))
