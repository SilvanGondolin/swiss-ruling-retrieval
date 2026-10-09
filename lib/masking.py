"""
Masking citation references out of query text before retrieval.

Cited articles/rulings appear directly in the case text - without masking, retrieval
collapses into citation lookup rather than testing semantic search. This
is a regex-based first pass covering the four leak surface forms
confirmed against real data: `<ref-law>`/`<ref-ruling>`
placeholder tags, literal "Art."/"art." statute references, BGE/ATF/DTF
ruling citations, and SR statute numbers.

This is not a substitute for span-level ground truth
(`rcds/swiss_citation_extraction`) - regexes cannot fully solve legal
citation extraction, as Stern et al. note directly: "Solving the task
with regexes is complicated due to extensive citation rules". Known gap:
chained letter-lists like "let. a et b LTF" only partially mask (the
trailing abbreviation can leak) - see `test_masking.py` for the exact
case.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_TAG_PATTERN = re.compile(r"<ref-law>|<ref-ruling>")

# "Art. 100 Abs. 1 BGG", "art. 108 al. 1 LTF", "Art. 44 - 48 BGG",
# "Art. 68 Abs. 1-3 BGG"
_ARTICLE_PATTERN = re.compile(
    r"\b[Aa]rt\.\s?\d+[a-z]?"
    r"(?:\s*(?:-|und|et|e)\s*\d+[a-z]?)?"
    r"(?:\s*(?:[Aa]bs\.?|[Aa]l\.|[Ll]et\.)\s*[\w.,\-]{0,20})*"
    r"(?:\s+[A-Z]{2,6}\b)?"
)

# "BGE 142 III 1", "ATF 1C_86/2015", "DTF ..."
_RULING_CITATION_PATTERN = re.compile(r"\b(?:BGE|ATF|DTF)\s?[\w./_-]+")

# "SR 173.110"
_SR_PATTERN = re.compile(r"\bSR\s?\d{3}(?:\.\d+)?")


def _replace_tag(match: re.Match) -> str:
    """
    Used as a `re.Pattern.sub`/`finditer` callback (see `_apply_passes`), which
    always calls it positionally - can't be made keyword-only, that's the `re`
    module's calling convention, not ours to change.

    :param match: A regex match of `_TAG_PATTERN`.
    :returns: "[LAW]" for a `<ref-law>` tag, "[RULING]" for `<ref-ruling>`.
    """
    return "[LAW]" if "law" in match.group(0) else "[RULING]"


@dataclass(frozen=True)
class Citation:
    """
    :ivar text: The original matched substring.
    :ivar label: "LAW" or "RULING".
    """

    text: str
    label: str


@dataclass(frozen=True)
class _Pass:
    pattern: re.Pattern
    replace: Callable[[re.Match], str]


@dataclass(frozen=True)
class _MaskingResult:
    masked_text: str
    citations: list[Citation]


_PASSES = [
    _Pass(pattern=_TAG_PATTERN, replace=_replace_tag),
    _Pass(pattern=_ARTICLE_PATTERN, replace=lambda _match: "[LAW]"),
    _Pass(pattern=_RULING_CITATION_PATTERN, replace=lambda _match: "[RULING]"),
    _Pass(pattern=_SR_PATTERN, replace=lambda _match: "[LAW]"),
]


def _apply_passes(*, text: str) -> _MaskingResult:
    """
    Run every masking pass over `text` in order, the shared logic behind both
    `mask_citations` and `find_citations`. Each pass only sees what's left after the
    previous one already replaced its matches, so a span consumed by an earlier pattern
    can't also be picked up by a later one.

    :param text: Raw case text.
    :returns: The fully masked text, and the citations found along the way, in the order
        they were matched.
    """
    found: list[Citation] = []
    for masking_pass in _PASSES:
        for match in masking_pass.pattern.finditer(text):
            label = "LAW" if masking_pass.replace(match) == "[LAW]" else "RULING"
            found.append(Citation(text=match.group(0), label=label))
        text = masking_pass.pattern.sub(masking_pass.replace, text)
    return _MaskingResult(masked_text=text, citations=found)


def mask_citations(*, text: str) -> str:
    """
    Replace citation references in `text` with `[LAW]`/`[RULING]` placeholders.

    :param text: Raw case text, e.g. a `Case.facts` or `Case.considerations` value.
    :returns:`text` with known citation surface forms replaced.
    """
    return _apply_passes(text=text).masked_text


def find_citations(*, text: str) -> list[Citation]:
    """
    Find the citation spans `mask_citations` would replace, without replacing them - for
    auditing what the masking function actually matches on real data.

    :param text: Raw case text, e.g. a `Case.facts` or `Case.considerations` value.
    :returns: The citations, in the order `mask_citations` would replace them.
    """
    return _apply_passes(text=text).citations
