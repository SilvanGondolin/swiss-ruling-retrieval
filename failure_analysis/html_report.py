"""Rendering retrieval outcomes as a single, self-contained static HTML page
- no server, no external resources, safe to open straight from disk.

The page is a table of cases. Clicking a row opens a side-by-side view of the case's
query and one of its rulings, built in the browser from a JSON payload that holds every
text once; Python only decides which words the two texts share."""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from html import escape

from failure_analysis.outcomes import CaseOutcome
from failure_analysis.worst_cases import WorstCases
from lib.language import Language
from lib.lexical.bm25 import DocumentFrequencies, stem_word
from lib.query import build_query
from lib.rulings import Ruling

# A masked-citation placeholder (see `lib.masking`), or a word as BM25 tokenizes it.
_WORD_OR_PLACEHOLDER = re.compile(r"\[(?:LAW|RULING)\]|\w\w+")

_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 95rem; margin: 2rem auto;
       padding: 0 1rem; color: #1a1a1a; }
h1 { font-size: 1.3rem; }
#summary { margin-bottom: 1rem; }
#controls { position: sticky; top: 0; z-index: 20; background: white; padding: 0.5rem 0;
            margin-bottom: 0.5rem; border-bottom: 1px solid #e4e4e4; }
#controls input[type=text], #controls select { font-size: 1rem; padding: 0.3rem; }
#controls label, #controls input, #controls select { margin-right: 0.8rem; }
#controls .rarity { white-space: nowrap; }
#controls #min-idf-value { display: inline-block; min-width: 2.2em; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #e4e4e4; }
th { cursor: pointer; background: #f5f5f5; }
tr.case-row { cursor: pointer; }
tr.case-row:hover { background: #f0f6ff; }
tr.case-row.open { background: #e6f0ff; }
tr.case-row.hidden { display: none; }
td.id { font-family: monospace; }
td.detail-cell { padding: 0.8rem 1rem 1.2rem; background: #fafafa; }
.picker { margin-bottom: 0.6rem; }
.picker select { font-size: 0.95rem; padding: 0.25rem; max-width: 100%; }
.hit { color: #1a7a1a; }
.miss { color: #8a5a00; }
.wrong { color: #b01a1a; }
.unresolvable { color: #888; }
.columns { display: flex; gap: 1rem; align-items: flex-start; }
.pane { flex: 1 1 0; min-width: 0; border: 1px solid #e4e4e4; border-radius: 4px;
        background: white; }
.pane h3 { margin: 0; padding: 0.4rem 0.7rem; font-size: 0.9rem; background: #f5f5f5;
           border-bottom: 1px solid #e4e4e4; }
.pane .text { padding: 0.6rem 0.7rem; max-height: 70vh; overflow-y: auto;
              white-space: pre-wrap; line-height: 1.45; font-size: 0.9rem; }
.pane .text { position: relative; }
.chunk { border-left: 4px solid #e4e4e4; padding-left: 0.6rem; margin: 0 0 0.8rem; }
.chunk.best { border-left-color: #2f7de1; background: #f3f8ff; }
.chunk-label { font-size: 0.75rem; color: #777; margin-bottom: 0.2rem; font-family: monospace; }
.chunk.best .chunk-label { color: #1c5bb0; font-weight: bold; }
.no-text { color: #888; font-style: italic; }
.seq { background: #ffd27a; border-radius: 2px; border-bottom: 2px solid #e09a00; }
.sequences { font-size: 0.85rem; margin: 0 0 0.6rem; color: #444; }
.sequences .seq-line { margin: 0.15rem 0; }
mark { background: #bde0ff; color: inherit; border-radius: 2px; padding: 0 1px; }
.citation { position: relative; padding: 0 2px; border-radius: 3px; cursor: help; }
.citation.all-hit { background: #c8f0c8; }
.citation.all-miss { background: #f8d8b0; }
.citation.mixed { background: #fff0b0; }
.citation .tooltip { display: none; position: absolute; left: 0; top: 1.4em; z-index: 10;
                      background: white; border: 1px solid #999; border-radius: 4px;
                      padding: 0.4rem 0.6rem; font-size: 0.85rem; white-space: nowrap;
                      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); }
.citation:hover .tooltip { display: block; }
.citation .tooltip > span { display: block; }
"""

_SCRIPT = r"""
const DATA = JSON.parse(document.getElementById('data').textContent);
const WORD = /\[(?:LAW|RULING)\]|[\p{L}\p{N}_]{2,}/gu;

function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function citation(kase) {
  const span = element('span', 'citation ' + kase.citationClass, '[RULING]');
  const tooltip = element('span', 'tooltip');
  for (const gold of kase.gold) {
    tooltip.append(element('span', gold.status, gold.id + ' - ' + gold.label));
  }
  span.append(tooltip);
  return span;
}

// `text` as DOM nodes with the words in `shared` (a Map from word to idf) marked;
// [RULING] placeholders become hoverable citations if `kase` is given.
function highlightWords(text, shared, kase) {
  const fragment = document.createDocumentFragment();
  let position = 0;
  for (const match of text.matchAll(WORD)) {
    fragment.append(text.slice(position, match.index));
    position = match.index + match[0].length;
    if (match[0] === '[RULING]' && kase) {
      fragment.append(citation(kase));
    } else if (shared.has(match[0].toLowerCase())) {
      const mark = element('mark', '', match[0]);
      mark.title = 'idf ' + shared.get(match[0].toLowerCase());
      fragment.append(mark);
    } else {
      fragment.append(match[0]);
    }
  }
  fragment.append(text.slice(position));
  return fragment;
}

// Like `highlightWords`, with the character `ranges` (sorted, disjoint, in `text`'s own
// coordinates) of shared sequences marked as a whole instead.
function highlight(text, shared, kase, ranges) {
  const fragment = document.createDocumentFragment();
  let position = 0;
  for (const range of ranges) {
    fragment.append(highlightWords(text.slice(position, range.start), shared, kase));
    const span = element('span', 'seq', text.slice(range.start, range.end));
    span.title = range.length + ' words in a row occur in both texts';
    fragment.append(span);
    position = range.end;
  }
  fragment.append(highlightWords(text.slice(position), shared, kase));
  return fragment;
}

// --- shared sequences: the longest runs of words that occur in both texts, found
// greedily: longest first, and no word is used by two sequences.

const WORD_IDS = new Map();
let placeholderCount = 0;

// The words of `text` as ids (the same word, in any case, always gets the same id) and
// their character spans. Every [LAW]/[RULING] placeholder gets an id of its own, so no
// sequence runs across one.
function wordTokens(text) {
  const tokens = {ids: [], starts: [], ends: []};
  for (const match of text.matchAll(WORD)) {
    let id;
    if (match[0].startsWith('[')) {
      placeholderCount += 1;
      id = -placeholderCount;
    } else {
      const word = match[0].toLowerCase();
      id = WORD_IDS.get(word);
      if (id === undefined) {
        id = WORD_IDS.size + 1;
        WORD_IDS.set(word, id);
      }
    }
    tokens.ids.push(id);
    tokens.starts.push(match.index);
    tokens.ends.push(match.index + match[0].length);
  }
  return tokens;
}

// A hash of every window of `length` consecutive ids, as a rolling polynomial hash.
function windowHashes(ids, length) {
  const count = ids.length - length + 1;
  if (count <= 0) return new Int32Array(0);
  const base = 1000003;
  let power = 1;
  for (let i = 1; i < length; i++) power = Math.imul(power, base);
  const hashes = new Int32Array(count);
  let hash = 0;
  for (let i = 0; i < length; i++) hash = (Math.imul(hash, base) + ids[i]) | 0;
  hashes[0] = hash;
  for (let i = 1; i < count; i++) {
    hash = (Math.imul(hash - Math.imul(ids[i - 1], power), base) + ids[i + length - 1]) | 0;
    hashes[i] = hash;
  }
  return hashes;
}

function hasCommonWindow(a, b, length) {
  const inB = new Set(windowHashes(b, length));
  return windowHashes(a, length).some(hash => inB.has(hash));
}

// prefix[i] = how many of the first i words are already used
function usedPrefix(used) {
  const prefix = new Int32Array(used.length + 1);
  for (let i = 0; i < used.length; i++) prefix[i + 1] = prefix[i] + used[i];
  return prefix;
}

function sameWords(a, i, b, j, length) {
  for (let k = 0; k < length; k++) if (a[i + k] !== b[j + k]) return false;
  return true;
}

function anyUsed(used, start, length) {
  for (let k = 0; k < length; k++) if (used[start + k]) return true;
  return false;
}

// The runs of at least `minLength` words that occur in both id lists, longest first and
// without overlapping each other, as {a, b, length}: start positions in each list.
function commonSequences(a, b, minLength) {
  const matches = [];
  let low = minLength;
  let high = Math.min(a.length, b.length);
  if (high < low || !hasCommonWindow(a, b, low)) return matches;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (hasCommonWindow(a, b, middle)) low = middle; else high = middle - 1;
  }
  const usedA = new Uint8Array(a.length);
  const usedB = new Uint8Array(b.length);
  for (let length = low; length >= minLength; length--) {
    const freeA = usedPrefix(usedA);
    const freeB = usedPrefix(usedB);
    const hashesA = windowHashes(a, length);
    const startsInB = new Map();
    windowHashes(b, length).forEach((hash, start) => {
      if (freeB[start + length] - freeB[start] !== 0) return;
      if (!startsInB.has(hash)) startsInB.set(hash, []);
      startsInB.get(hash).push(start);
    });
    hashesA.forEach((hash, start) => {
      if (freeA[start + length] - freeA[start] !== 0 || anyUsed(usedA, start, length)) return;
      for (const other of startsInB.get(hash) || []) {
        if (anyUsed(usedB, other, length) || !sameWords(a, start, b, other, length)) continue;
        usedA.fill(1, start, start + length);
        usedB.fill(1, other, other + length);
        matches.push({a: start, b: other, length});
        break;
      }
    });
  }
  return matches;
}

// The character ranges `matches` cover in a text with `tokens`, taking each match's
// start position in that text from `startOf`, sorted by position.
function sequenceRanges(tokens, matches, startOf) {
  return matches
    .map(m => ({
      start: tokens.starts[startOf(m)],
      end: tokens.ends[startOf(m) + m.length - 1],
      length: m.length,
    }))
    .sort((x, y) => x.start - y.start);
}

// `ranges` restricted to the window [from, to) of the text they refer to, shifted so
// they refer to that window's own coordinates.
function clipRanges(ranges, from, to) {
  return ranges
    .filter(range => range.end > from && range.start < to)
    .map(range => ({
      start: Math.max(range.start, from) - from,
      end: Math.min(range.end, to) - from,
      length: range.length,
    }));
}

const RULING_CACHE = new Map();

// A ruling's chunks joined into one text (a space between chunks, which `offsets`
// accounts for), its words, and where each chunk starts in the joined text.
function rulingText(id) {
  if (!RULING_CACHE.has(id)) {
    const chunks = DATA.rulings[id];
    const offsets = [];
    let offset = 0;
    for (const chunk of chunks) {
      offsets.push(offset);
      offset += chunk.length + 1;
    }
    const text = chunks.join(' ');
    RULING_CACHE.set(id, {text, offsets, tokens: wordTokens(text)});
  }
  return RULING_CACHE.get(id);
}

// The words `option` shares with the query that the rarity controls leave highlighted.
function shownWords(option) {
  const minIdf = Number(document.getElementById('min-idf').value);
  const skipNumbers = document.getElementById('ignore-numbers').checked;
  const shown = new Map();
  for (const [word, idf] of Object.entries(option.sharedWords)) {
    if (idf >= minIdf && !(skipNumbers && /^\d+$/.test(word))) shown.set(word, idf);
  }
  return shown;
}

function pane(title, titleClass, body) {
  const container = element('div', 'pane');
  container.append(element('h3', titleClass, title), body);
  return container;
}

function showOption(kase, columns, info, option) {
  const shared = shownWords(option);
  const chunks = option.hasRuling ? DATA.rulings[option.id] : [];
  let queryRanges = [];
  let rulingRanges = [];
  info.replaceChildren();
  if (option.hasRuling && document.getElementById('show-sequences').checked) {
    const minLength = Number(document.getElementById('min-seq').value);
    if (!kase.tokens) kase.tokens = wordTokens(kase.query);
    const joined = rulingText(option.id);
    const matches = commonSequences(kase.tokens.ids, joined.tokens.ids, minLength);
    queryRanges = sequenceRanges(kase.tokens, matches, m => m.a);
    rulingRanges = sequenceRanges(joined.tokens, matches, m => m.b);
    info.append(element('div', '', matches.length + ' shared sequences of at least '
      + minLength + ' words (orange); longest first:'));
    [...queryRanges].sort((x, y) => y.length - x.length).slice(0, 3).forEach(range => {
      info.append(element('div', 'seq-line', range.length + ' words: '
        + kase.query.slice(range.start, range.end)));
    });
  }
  const query = element('div', 'text');
  query.append(highlight(kase.query, shared, kase, queryRanges));
  const ruling = element('div', 'text');
  let bestChunk = null;
  if (chunks.join('')) {
    if (option.chunkScores) {
      chunks.forEach((chunk, index) => {
        const best = index === option.bestChunk;
        const box = element('div', best ? 'chunk best' : 'chunk');
        const score = Number(option.chunkScores[index].toPrecision(4));
        const label = 'chunk ' + (index + 1) + '/' + chunks.length + ' - score ' + score
          + (best ? ' - best match: the ruling is ranked by this score' : '');
        const from = rulingText(option.id).offsets[index];
        box.append(
          element('div', 'chunk-label', label),
          highlight(chunk, shared, null, clipRanges(rulingRanges, from, from + chunk.length)),
        );
        ruling.append(box);
        if (best) bestChunk = box;
      });
    } else {
      ruling.append(highlight(chunks.join(' '), shared, null, rulingRanges));
    }
  } else {
    const note = option.hasRuling ? '(no text)' : '(not in the corpus: unresolvable)';
    ruling.append(element('span', 'no-text', note));
  }
  const where = bestChunk ? ', best chunk ' + (option.bestChunk + 1) + '/' + chunks.length : '';
  const title = option.id + ' - ' + option.label + ' (' + shared.size
    + ' shared words shown' + where + ')';
  columns.replaceChildren(pane('Query', '', query), pane(title, option.status, ruling));
  if (bestChunk) ruling.scrollTop = bestChunk.offsetTop - 8;
}

function buildDetail(kase) {
  const row = element('tr', 'detail');
  const picker = element('div', 'picker', 'Ruling: ');
  const select = element('select');
  kase.options.forEach((option, index) => {
    const tag = element('option', option.status, option.name + ' - ' + option.label);
    tag.value = index;
    tag.disabled = !option.hasRuling;
    select.append(tag);
  });
  const columns = element('div', 'columns');
  const info = element('div', 'sequences');
  const choose = () => {
    const option = kase.options[select.value];
    select.className = option.status;
    showOption(kase, columns, info, option);
  };
  select.onchange = choose;
  const firstUsable = kase.options.findIndex(option => option.hasRuling);
  if (firstUsable < 0) {
    columns.append(element('span', 'no-text', 'No ruling of this case is in the corpus.'));
  } else {
    select.value = firstUsable;
    row.refresh = choose;
    choose();
  }
  picker.append(select);
  const cell = element('td', 'detail-cell');
  cell.colSpan = document.querySelectorAll('thead th').length;
  cell.append(picker, info, columns);
  row.append(cell);
  return row;
}

function toggleCase(row) {
  const next = row.nextElementSibling;
  if (next && next.classList.contains('detail')) {
    next.remove();
    row.classList.remove('open');
    return;
  }
  row.after(buildDetail(DATA.cases[row.dataset.index]));
  row.classList.add('open');
}

function refreshOpen() {
  const slider = document.getElementById('min-idf');
  document.getElementById('min-idf-value').textContent = Number(slider.value).toFixed(1);
  document.getElementById('min-seq-value').textContent = document.getElementById('min-seq').value;
  for (const detail of document.querySelectorAll('tr.detail')) {
    if (detail.refresh) detail.refresh();
  }
}

function closeAll() {
  for (const detail of document.querySelectorAll('tr.detail')) detail.remove();
  for (const row of document.querySelectorAll('tr.open')) row.classList.remove('open');
}

function applyFilters() {
  const q = document.getElementById('search').value.toLowerCase();
  const lang = document.getElementById('language').value;
  const status = document.getElementById('status').value;
  for (const row of document.querySelectorAll('tr.case-row')) {
    const matchesText = !q || row.dataset.search.includes(q);
    const matchesLang = !lang || row.dataset.language === lang;
    const matchesStatus = !status || row.dataset.statuses.split(' ').includes(status);
    const visible = matchesText && matchesLang && matchesStatus;
    row.classList.toggle('hidden', !visible);
    const next = row.nextElementSibling;
    if (!visible && next && next.classList.contains('detail')) toggleCase(row);
  }
}

function sortBy(column) {
  closeAll();
  const body = document.querySelector('tbody');
  const header = document.querySelectorAll('thead th')[column];
  const direction = header.dataset.direction === 'asc' ? -1 : 1;
  header.dataset.direction = direction === 1 ? 'asc' : 'desc';
  const numeric = header.dataset.numeric === 'true';
  const key = row => numeric ? Number(row.children[column].dataset.sort)
                             : row.children[column].dataset.sort;
  const rows = [...body.children].sort(
    (a, b) => direction * (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0)
  );
  body.append(...rows);
}
"""


class RulingStatus(StrEnum):
    """How a ruling relates to a case's retrieval; also its CSS class."""

    HIT = "hit"
    MISS = "miss"
    WRONG = "wrong"
    UNRESOLVABLE = "unresolvable"


_STATUS_LABELS = {
    RulingStatus.HIT: "retrieved",
    RulingStatus.MISS: "missed",
    RulingStatus.WRONG: "top prediction, not cited",
    RulingStatus.UNRESOLVABLE: "not in corpus",
}


@dataclass(frozen=True)
class RulingOption:
    """
    One entry of a case's ruling dropdown.

    :ivar ruling_id: The ruling's decision_id.
    :ivar ruling: The ruling; None if it isn't in the corpus.
    :ivar status: How the ruling relates to the case.
    """

    ruling_id: str
    ruling: Ruling | None
    status: RulingStatus

    @property
    def name(self) -> str:
        """The ruling's file name if it has one, else its decision_id."""
        if self.ruling is not None and self.ruling.file_name:
            return self.ruling.file_name
        return self.ruling_id


@dataclass(frozen=True)
class SharedWords:
    """
    The vocabulary a query and a ruling have in common.

    :ivar stems: The stems of the non-stopword words occurring in both.
    :ivar idf_by_word: Every lowercased word form of either text whose stem is in
        `stems`, with its stem's rarity (see
        `lib.lexical.bm25.DocumentFrequencies.idf`).
    """

    stems: frozenset[str]
    idf_by_word: dict[str, float]


@dataclass(frozen=True)
class Column:
    """
    :ivar name: The table header.
    :ivar numeric: Whether the column sorts as numbers rather than text.
    """

    name: str
    numeric: bool


_COLUMNS = [
    Column(name="case", numeric=False),
    Column(name="language", numeric=False),
    Column(name="law area", numeric=False),
    Column(name="recall", numeric=True),
    Column(name="hits", numeric=True),
    Column(name="cited", numeric=True),
    Column(name="top-1 cited", numeric=False),
]


def _ruling_options(
    *, outcome: CaseOutcome, rulings_by_id: dict[str, Ruling]
) -> list[RulingOption]:
    """
    :param outcome: The case outcome to list rulings for.
    :param rulings_by_id: Every gold ruling and top prediction `outcome` has, keyed by
        decision_id.
    :returns: The case's gold rulings - retrieved ones first, then missed ones, then
        those missing from the corpus - followed by the top-ranked prediction if it
        isn't one of the gold rulings.
    """
    options: list[RulingOption] = []
    for ruling_id in sorted(outcome.hits):
        options.append(
            RulingOption(
                ruling_id=ruling_id,
                ruling=rulings_by_id[ruling_id],
                status=RulingStatus.HIT,
            )
        )
    unresolvable: list[RulingOption] = []
    for ruling_id in sorted(outcome.misses):
        ruling = rulings_by_id.get(ruling_id)
        if ruling is None:
            unresolvable.append(
                RulingOption(
                    ruling_id=ruling_id,
                    ruling=None,
                    status=RulingStatus.UNRESOLVABLE,
                )
            )
        else:
            options.append(
                RulingOption(
                    ruling_id=ruling_id, ruling=ruling, status=RulingStatus.MISS
                )
            )
    options.extend(unresolvable)

    top_prediction = outcome.retrieved[0] if outcome.retrieved else None
    if top_prediction is not None and top_prediction not in outcome.gold:
        options.append(
            RulingOption(
                ruling_id=top_prediction,
                ruling=rulings_by_id[top_prediction],
                status=RulingStatus.WRONG,
            )
        )
    return options


def _words_by_stem(*, text: str, language: Language) -> dict[str, set[str]]:
    """
    :param text: Query or ruling text.
    :param language: The text's language.
    :returns: Each stem of its non-stopword words (placeholders like [LAW] ignored) with
        the lowercased word forms that produce it.
    """
    words_by_stem: dict[str, set[str]] = {}
    for match in _WORD_OR_PLACEHOLDER.finditer(text):
        word = match.group()
        if word.startswith("["):
            continue
        stem = stem_word(word, language)
        if stem is not None:
            words_by_stem.setdefault(stem, set()).add(word.lower())
    return words_by_stem


def _shared_words(
    *,
    query: str,
    query_language: Language,
    ruling: Ruling,
    frequencies: DocumentFrequencies,
) -> SharedWords:
    """
    :param query: The case's citation-masked query text.
    :param query_language: The query's language.
    :param ruling: The ruling to compare it with.
    :param frequencies: Document frequencies to rate each shared stem's rarity with,
        as a stem of the query's language.
    :returns: The stems occurring in both texts, with the word forms of either text
        that have one of those stems and their rarity.
    """
    query_words = _words_by_stem(text=query, language=query_language)
    ruling_words = _words_by_stem(text=ruling.considerations, language=ruling.language)
    stems = frozenset(query_words) & frozenset(ruling_words)
    idf_by_word = {
        word: frequencies.idf(stem=stem, language=query_language)
        for stem in stems
        for word in query_words[stem] | ruling_words[stem]
    }
    return SharedWords(stems=stems, idf_by_word=idf_by_word)


def _citation_overall_class(*, outcome: CaseOutcome) -> str:
    """
    :param outcome: The case outcome to classify.
    :returns: "all-hit" if every gold ruling was retrieved, "all-miss" if none were,
        "mixed" otherwise - the [RULING] highlight's color. A `<ref-ruling>` tag - one
        of the four citation forms `[RULING]` replaces (see `lib.masking`) - carries
        no identifying text in the source data, so a specific mention can't be
        attributed to one gold ruling: every mention in a case gets this same,
        case-level color. [LAW] is left as plain text: no method in this project
        retrieves laws.
    """
    if not outcome.misses:
        return "all-hit"
    if not outcome.hits:
        return "all-miss"
    return "mixed"


def _rounded(scores: list[float] | None) -> list[float] | None:
    """
    :param scores: Chunk scores, or None.
    :returns: The scores rounded for the page, or None.
    """
    return None if scores is None else [round(score, 4) for score in scores]


def _best_chunk(scores: list[float] | None) -> int | None:
    """
    :param scores: A ruling's chunk scores, or None.
    :returns: The index of the highest-scoring chunk - the one that gives the ruling its
        score; the first one on a tie. None without scores.
    """
    return None if not scores else max(range(len(scores)), key=scores.__getitem__)


def _case_payload(
    *,
    outcome: CaseOutcome,
    options: list[RulingOption],
    frequencies: DocumentFrequencies,
    chunk_scores: dict[str, list[float]],
) -> dict[str, object]:
    """
    :param outcome: The case outcome to describe.
    :param options: The case's dropdown options, see `_ruling_options`.
    :param frequencies: Document frequencies to rate shared words' rarity with.
    :param chunk_scores: The method's score for each chunk of each ruling the case
        shows, keyed by decision_id; empty if the method doesn't score chunks.
    :returns: The JSON-serializable data the page needs to open the case: its query,
        its gold rulings for the [RULING] tooltips, and one entry per dropdown option
        with the words that option shares with the query.
    """
    case = outcome.case
    query = build_query(case=case)
    option_payloads: list[dict[str, object]] = []
    for option in options:
        shared = (
            _shared_words(
                query=query,
                query_language=case.language,
                ruling=option.ruling,
                frequencies=frequencies,
            )
            if option.ruling is not None
            else SharedWords(stems=frozenset(), idf_by_word={})
        )
        option_payloads.append(
            {
                "id": option.ruling_id,
                "name": option.name,
                "status": option.status.value,
                "label": _STATUS_LABELS[option.status],
                "hasRuling": option.ruling is not None,
                "chunkScores": _rounded(chunk_scores.get(option.ruling_id)),
                "bestChunk": _best_chunk(chunk_scores.get(option.ruling_id)),
                "sharedWords": {
                    word: round(idf, 2)
                    for word, idf in sorted(shared.idf_by_word.items())
                },
            }
        )
    gold = [
        {
            "id": option.ruling_id,
            "status": option.status.value,
            "label": _STATUS_LABELS[option.status],
        }
        for option in options
        if option.status is not RulingStatus.WRONG
    ]
    return {
        "query": query,
        "citationClass": _citation_overall_class(outcome=outcome),
        "gold": gold,
        "options": option_payloads,
    }


def _statuses(*, options: list[RulingOption]) -> str:
    """
    :param options: A case's dropdown options.
    :returns: The distinct filter groups the case has a ruling in, space-separated:
        "hit", "miss" (missed rulings, including those absent from the corpus) and
        "wrong".
    """
    groups = {
        "miss" if option.status is RulingStatus.UNRESOLVABLE else option.status.value
        for option in options
    }
    return " ".join(sorted(groups))


def _render_row(
    *, index: int, outcome: CaseOutcome, options: list[RulingOption]
) -> str:
    """
    :param index: The case's position in the payload's `cases`.
    :param outcome: The case outcome to summarize.
    :param options: The case's dropdown options, see `_ruling_options`.
    :returns: A clickable `<tr>` summarizing `outcome`; sortable by every column.
    """
    case = outcome.case
    top_is_cited = bool(outcome.retrieved) and outcome.retrieved[0] in outcome.gold
    top_label = "yes" if top_is_cited else "no" if outcome.retrieved else "-"
    search = " ".join(
        [case.decision_id, case.language, case.law_area or ""]
        + [f"{option.ruling_id} {option.name}" for option in options]
    ).lower()
    law_area = escape(case.law_area) if case.law_area else "(none)"
    return (
        f"<tr class='case-row' data-index='{index}'"
        f" data-language='{escape(case.language)}'"
        f" data-statuses='{_statuses(options=options)}'"
        f" data-search='{escape(search)}' onclick='toggleCase(this)'>"
        f"<td class='id' data-sort='{escape(case.decision_id)}'>"
        f"{escape(case.decision_id)}</td>"
        f"<td data-sort='{escape(case.language)}'>{escape(case.language)}</td>"
        f"<td data-sort='{law_area}'>{law_area}</td>"
        f"<td data-sort='{outcome.recall}'>{outcome.recall:.2f}</td>"
        f"<td data-sort='{len(outcome.hits)}'>{len(outcome.hits)}</td>"
        f"<td data-sort='{len(outcome.gold)}'>{len(outcome.gold)}</td>"
        f"<td data-sort='{top_label}'>{top_label}</td>"
        "</tr>"
    )


def _payload_json(
    *,
    worst_cases: WorstCases,
    options_by_case: list[list[RulingOption]],
    frequencies: DocumentFrequencies,
) -> str:
    """
    :param worst_cases: The cases to describe, with the chunk data of chunk-scoring
        methods.
    :param options_by_case: Each case's dropdown options, aligned with the outcomes.
    :param frequencies: Document frequencies to rate shared words' rarity with.
    :returns: The JSON the page's script reads, made safe to embed in a `<script>`
        element: every case once and every ruling text once, however many cases show
        it.
    """
    cases = [
        _case_payload(
            outcome=outcome,
            options=options,
            frequencies=frequencies,
            chunk_scores=worst_cases.chunk_scores.get(outcome.case.decision_id, {}),
        )
        for outcome, options in zip(worst_cases.outcomes, options_by_case, strict=True)
    ]
    # each ruling as a list of chunks: the method's own for chunk-scoring methods, else
    # the whole text as one
    rulings = {
        option.ruling_id: worst_cases.chunks_by_id.get(
            option.ruling_id, [option.ruling.considerations]
        )
        for options in options_by_case
        for option in options
        if option.ruling is not None
    }
    payload = json.dumps({"cases": cases, "rulings": rulings}, ensure_ascii=False)
    return payload.replace("<", "\\u003c")


def render_report(
    *, worst_cases: WorstCases, method_label: str, frequencies: DocumentFrequencies
) -> str:
    """
    :param worst_cases: The cases to render, and the ruling text they need.
    :param method_label: The retrieval method's name, shown in the report's heading.
    :param frequencies: Document frequencies over the ruling corpus, to rate how rare
        each word the query and a ruling share is.
    :returns: A self-contained HTML page: a filterable, sortable table of the cases,
        each opening a side-by-side view of its query and a ruling.
    """
    outcomes = worst_cases.outcomes
    options_by_case = [
        _ruling_options(outcome=outcome, rulings_by_id=worst_cases.rulings_by_id)
        for outcome in outcomes
    ]
    languages = sorted({outcome.case.language for outcome in outcomes})
    language_options = "".join(
        f"<option value='{escape(lang)}'>{escape(lang)}</option>" for lang in languages
    )
    header = "".join(
        f"<th data-numeric='{str(column.numeric).lower()}' onclick='sortBy({index})'>"
        f"{column.name}</th>"
        for index, column in enumerate(_COLUMNS)
    )
    rows = "\n".join(
        _render_row(index=index, outcome=outcome, options=options)
        for index, (outcome, options) in enumerate(
            zip(outcomes, options_by_case, strict=True)
        )
    )
    payload = _payload_json(
        worst_cases=worst_cases,
        options_by_case=options_by_case,
        frequencies=frequencies,
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Failure report: {escape(method_label)}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>Failure report: {escape(method_label)}</h1>
<div id="summary">{worst_cases.val_case_count} cases evaluated, showing
  {len(outcomes)} (worst first). Click a case to compare its query with a
  ruling. Blue words occur in both and are at least as rare as the rarity slider's
  setting (hover one for its idf); orange runs are word sequences that occur in both.</div>
<div id="controls">
  <input id="search" type="text" placeholder="search case or ruling id"
         oninput="applyFilters()">
  <select id="language" onchange="applyFilters()">
    <option value="">all languages</option>
    {language_options}
  </select>
  <select id="status" onchange="applyFilters()">
    <option value="">all cases</option>
    <option value="hit">with a correct prediction</option>
    <option value="miss">with a missed ruling</option>
    <option value="wrong">with a false positive</option>
  </select>
  <span class="rarity">
    <label>min rarity (idf)
      <input id="min-idf" type="range" min="0" max="10" step="0.5" value="3"
             oninput="refreshOpen()"></label>
    <span id="min-idf-value">3.0</span>
  </span>
  <label><input id="ignore-numbers" type="checkbox" checked
                onchange="refreshOpen()"> ignore numbers</label>
  <span class="rarity">
    <label><input id="show-sequences" type="checkbox" checked
                  onchange="refreshOpen()"> shared sequences of at least</label>
    <input id="min-seq" type="range" min="2" max="15" step="1" value="4"
           oninput="refreshOpen()">
    <span id="min-seq-value">4</span> words
  </span>
</div>
<table>
<thead><tr>{header}</tr></thead>
<tbody>
{rows}
</tbody>
</table>
<script id="data" type="application/json">{payload}</script>
<script>{_SCRIPT}</script>
</body>
</html>"""
