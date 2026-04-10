"""
translator.py – Auto-translation layer for the Bible Code ELS searcher.

Converts English (or any non-Hebrew) search terms to Hebrew consonants ready
for ELSEngine.search(), with optional synonym expansion.

Public API
----------
prepare_search_terms(inputs, expand=False, expand_count=2)
    Accepts a list of strings that may be English, English+Hebrew mixed, or
    purely Hebrew.  Returns a ``PreparedTerms`` object containing:

    - ``words``   : list[str]  — Hebrew Unicode strings to search (consonants only)
    - ``labels``  : list[str]  — human-readable labels preserving the original English
    - ``origins`` : dict[str, str] — maps each Hebrew word → its origin label

    When *expand* is True, each translated term gains up to *expand_count*
    Hebrew synonyms sourced via Google Translate paraphrase queries.

    Results are cached in ``data/translation_cache.json`` to avoid repeated
    network calls.

Detection
---------
A string is considered "Hebrew" if it contains at least one character in the
Hebrew Unicode block (U+05D0–U+05EA).  All other strings are treated as English
and sent to the Google Translate API (source='en', target='iw').

Normalisation
-------------
All translated Hebrew strings are passed through the same final-form
normalisation used by data_loader.py (_FINAL_TO_NONFINAL) and filtered to
the 22-consonant alphabet before being returned.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import expander as _expander

# ── Hebrew helpers (mirrors data_loader.py) ───────────────────────────────────

_HEBREW_RE = re.compile(r"[\u05D0-\u05EA]")

_FINAL_TO_NONFINAL = str.maketrans(
    {
        "\u05DA": "\u05DB",  # ך final kaf   → כ
        "\u05DD": "\u05DE",  # ם final mem   → מ
        "\u05DF": "\u05E0",  # ן final nun   → נ
        "\u05E3": "\u05E4",  # ף final pe    → פ
        "\u05E5": "\u05E6",  # ץ final tsade → צ
    }
)

# Hebrew Unicode consonant range (alef–tav, non-final only after normalisation)
_HEB_CONSONANTS = set(chr(c) for c in range(0x05D0, 0x05EB))


def _is_hebrew_input(text: str) -> bool:
    """Return True if *text* contains at least one Hebrew letter."""
    return bool(_HEBREW_RE.search(text))


def _normalise_hebrew(text: str) -> str:
    """
    Normalise a raw Hebrew string:
    1. Strip nikud / cantillation (non-spacing combining marks).
    2. Map final-form letters to their non-final equivalents.
    3. Keep only the 22 consonants.
    """
    # Strip all Unicode combining characters (nikud, cantillation)
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    normalised = stripped.translate(_FINAL_TO_NONFINAL)
    return "".join(ch for ch in normalised if ch in _HEB_CONSONANTS)


# ── Gregorian-to-Hebrew year map ─────────────────────────────────────────────

YEAR_MAP: dict[str, str] = {
    "2024": "תשפד",
    "2025": "תשפה",
    "2026": "תשפו",
    "2027": "תשפז",
    "2028": "תשפח",
    "2029": "תשפט",
    "2030": "תשצ",
    "2031": "תשצא",
    "2032": "תשצב",
    "2033": "תשצג",
    "2034": "תשצד",
    "2035": "תשצה",
    "2036": "תשצו",
    "2037": "תשצז",
    "2038": "תשצח",
    "2039": "תשצט",
    "2040": "תת",
}

# ── Hebrew calendar conversion utilities ─────────────────────────────────────

# Matches ISO (2026-08-14), slash (14/8/2026) or month-name patterns.
# Requires at least one digit adjacent to a separator/month-name so bare
# English words ("May", "March") or plain years ("2030") are not caught here.
_DATE_PATTERN = re.compile(
    r"""
    \d{1,4}[-/]\d{1,2}[-/]\d{1,4}                        # ISO/numeric separators
    |
    \d{1,2}\s+                                            # day number + space before month
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
       Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
    |
    (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
       Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
    \s+\d{1,2}                                            # month name followed by day number
    """,
    re.VERBOSE | re.IGNORECASE,
)

_HEBREW_MONTH_NAMES: dict[int, str] = {
    1: "ניסן",  2: "אייר",  3: "סיון",  4: "תמוז",
    5: "אב",    6: "אלול",  7: "תשרי",  8: "חשון",
    9: "כסלו",  10: "טבת",  11: "שבט",  12: "אדר",  13: "אדר ב",
}

# Gematria ordered largest-first for greedy subtraction
_GEMATRIA_TABLE: list[tuple[int, str]] = [
    (400, "ת"), (300, "ש"), (200, "ר"), (100, "ק"),
    (90, "צ"),  (80, "פ"),  (70, "ע"),  (60, "ס"),  (50, "נ"),
    (40, "מ"),  (30, "ל"),  (20, "כ"),  (10, "י"),
    (9, "ט"),   (8, "ח"),   (7, "ז"),   (6, "ו"),   (5, "ה"),
    (4, "ד"),   (3, "ג"),   (2, "ב"),   (1, "א"),
]


def _num_to_gematria(n: int) -> str:
    """Convert a positive integer to a Hebrew gematria string."""
    result: list[str] = []
    for val, letter in _GEMATRIA_TABLE:
        while n >= val:
            result.append(letter)
            n -= val
    return "".join(result)


def _day_to_gematria(day: int) -> str:
    """Day gematria with the standard 15→טו and 16→טז exceptions (avoids divine names)."""
    if day == 15:
        return "טו"
    if day == 16:
        return "טז"
    return _num_to_gematria(day)


def _hebrew_year_short(gregorian_year: int) -> str:
    """Return the short-form Hebrew gematria for a Gregorian year.

    Looks up YEAR_MAP first for the project's canonical spellings;
    falls back to dynamically computing Hebrew year = Gregorian + 3760
    (the convention used throughout this project for year-long spans).
    Short form drops the thousands digit (5000 → ה is omitted).
    """
    cached = YEAR_MAP.get(str(gregorian_year))
    if cached:
        return cached
    return _num_to_gematria((gregorian_year + 3760) % 1000)


def gregorian_to_hebrew_tokens(date_str: str) -> tuple[str, str, str]:
    """Convert a Gregorian date string to Hebrew ELS search tokens.

    Parameters
    ----------
    date_str:
        Any human-readable date, e.g. ``"August 14 2026"``, ``"2026-08-14"``,
        ``"14/8/2026"``.

    Returns
    -------
    spaced : str
        Human-readable Hebrew date, e.g. ``"יד אב תשפו"``.
    condensed : str
        Space-stripped version for ELS search, e.g. ``"ידאבתשפו"``.
    label : str
        Display label for the results table.

    Raises
    ------
    ImportError
        If ``pyluach`` or ``python-dateutil`` are not installed.
    ValueError
        If the date string cannot be parsed.
    """
    try:
        from dateutil import parser as _du  # lazy import
    except ImportError as exc:
        raise ImportError("python-dateutil is required: uv add python-dateutil") from exc
    try:
        from pyluach import dates as _pyd  # lazy import
    except ImportError as exc:
        raise ImportError("pyluach is required: uv add pyluach") from exc

    g = _du.parse(date_str, dayfirst=False)
    h = _pyd.GregorianDate(g.year, g.month, g.day).to_heb()

    day_heb = _day_to_gematria(h.day)
    month_heb = _HEBREW_MONTH_NAMES.get(h.month, "?")
    year_heb = _num_to_gematria(h.year % 1000)

    spaced = f"{day_heb} {month_heb} {year_heb}"
    condensed = f"{day_heb}{month_heb}{year_heb}"
    label = (
        f"{date_str} → {spaced}  "
        f"[{h.day} {_HEBREW_MONTH_NAMES.get(h.month, '?')} {h.year}]"
    )
    return spaced, condensed, label


# ── Caching ───────────────────────────────────────────────────────────────────

_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "translation_cache.json"


def _load_cache() -> dict[str, str]:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Translation ───────────────────────────────────────────────────────────────

def _translate_to_hebrew(text: str, cache: dict[str, str]) -> str:
    """
    Translate *text* to Hebrew via Google Translate (deep-translator).

    Returns the normalised consonant-only Hebrew string.
    Raises ``ValueError`` if the result contains no Hebrew letters.
    """
    key = f"en:iw:{text.strip().lower()}"
    if key in cache:
        raw = cache[key]
    else:
        from deep_translator import GoogleTranslator  # lazy import
        raw = GoogleTranslator(source="en", target="iw").translate(text)
        cache[key] = raw
        _save_cache(cache)

    result = _normalise_hebrew(raw)
    if not result:
        raise ValueError(
            f"Translation of {text!r} returned no Hebrew characters. "
            f"Raw API response: {raw!r}  — supply the Hebrew directly instead."
        )
    return result


def _fetch_synonyms(
    english_term: str,
    base_hebrew: str,
    count: int,
    cache: dict[str, str],
    method: Literal["lexicon", "llm"] = "lexicon",
) -> list[str]:
    """
    Obtain up to *count* Hebrew synonyms for *english_term*.

    method="lexicon" (default, recommended)
        Check ``expander.ENGLISH_TO_HEBREW`` first (static, no network).
        Returns synonyms from the lexicon; never raises; always deterministic.

    method="llm"
        Legacy behaviour: translate paraphrase probes via Google Translate.
        Requires a network connection; prone to returning long multi-word
        phrases that produce 0 ELS matches.  Kept for backward-compatibility.

    Returns a deduplicated list of Hebrew consonant strings distinct from
    *base_hebrew*.  May return fewer than *count* entries; never raises.
    """
    if method == "lexicon":
        # Static lexicon lookup — fast, deterministic, no network required
        candidates = _expander.get_synonyms(english_term)
        # get_synonyms returns Hebrew strings; filter out the base translation
        synonyms = [h for h in candidates if h != base_hebrew]
        return synonyms[:count]

    # method == "llm" — legacy Google Translate paraphrase probes
    synonyms: list[str] = []
    probes = [
        f"synonym of {english_term}",
        f"another word for {english_term}",
        f"alternative term for {english_term}",
    ]
    for probe in probes:
        if len(synonyms) >= count:
            break
        try:
            candidate = _translate_to_hebrew(probe, cache)
            if candidate and candidate != base_hebrew and candidate not in synonyms:
                synonyms.append(candidate)
        except (ValueError, Exception):  # network errors, bad responses
            continue
    return synonyms[:count]


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class PreparedTerms:
    """
    Result of ``prepare_search_terms()``.

    Attributes
    ----------
    words : list[str]
        Hebrew Unicode consonant strings, ready for ``ELSEngine.search()``.
    labels : list[str]
        Human-readable label for each word in *words*.  For directly-supplied
        Hebrew the label equals the word itself.  For translated terms the label
        is ``"english_input → hebrewword"`` so the output table stays readable.
    origins : dict[str, str]
        Maps each Hebrew word in *words* → its label.  Convenience accessor for
        column-labelling in the results table.
    """

    words: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    origins: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.words)

    def __iter__(self):
        return iter(self.words)


def prepare_search_terms(
    inputs: Sequence[str],
    *,
    expand: bool = False,
    expand_count: int = 2,
    expand_method: Literal["lexicon", "llm"] = "lexicon",
    show_translations: bool = True,
    translate_english: bool = True,
) -> PreparedTerms:
    """
    Convert a list of search term strings to normalised Hebrew, translating
    English inputs automatically.

    Parameters
    ----------
    translate_english :
        When True (default), send unrecognised English strings to Google
        Translate.  Set to False to skip translation while still retaining
        automatic year and date conversion — useful when ``--translate`` is
        absent but date/year detection should remain active.

    Parameters
    ----------
    inputs :
        Strings from ``--words``.  May be Hebrew, English, or mixed.
    expand :
        When True, add up to *expand_count* Hebrew synonyms per term.
    expand_count :
        Max synonyms to add per translated English term (default 2).
    expand_method :
        ``"lexicon"`` (default) — use the static ``expander.py`` dictionary;
        deterministic, no network required, themed around Biblical concepts.
        ``"llm"`` — legacy Google Translate paraphrase probes (unreliable).
    show_translations :
        When True, print each translation/synonym resolution to stdout so the
        user can confirm before the search begins.

    Returns
    -------
    PreparedTerms
    """
    cache = _load_cache()
    result = PreparedTerms()

    for raw in inputs:
        raw = raw.strip()
        if not raw:
            continue

        if _is_hebrew_input(raw):
            # Already Hebrew — normalise and use directly
            hebrew = _normalise_hebrew(raw)
            if not hebrew:
                raise ValueError(
                    f"Input {raw!r} appears Hebrew but contains no consonants after normalisation."
                )
            label = hebrew
            if show_translations and hebrew != raw:
                print(f"  Normalised: {raw!r} → {hebrew!r}")
            result.words.append(hebrew)
            result.labels.append(label)
            result.origins[hebrew] = label

            # Hebrew direct input: also look up related roots in the lexicon
            if expand:
                synonyms = _expander.get_synonyms(hebrew, normalise_fn=_normalise_hebrew, count=expand_count)
                for syn in synonyms:
                    syn_label = f"{hebrew} (related) → {syn}"
                    if show_translations:
                        print(f"  Related:    {hebrew!r} → {syn!r}")
                    result.words.append(syn)
                    result.labels.append(syn_label)
                    result.origins[syn] = syn_label
        elif raw.lstrip("-").isdigit():
            # Purely numeric — year conversion (YEAR_MAP first, then dynamic).
            digits = raw.lstrip("-")
            if len(digits) == 4:
                hebrew = _hebrew_year_short(int(digits))
                label = f"{raw} → {hebrew}"
                if show_translations:
                    print(f"  Detected Year: {raw!r} → {hebrew!r}")
                result.words.append(hebrew)
                result.labels.append(label)
                result.origins[hebrew] = label
            else:
                print(
                    f"  Skipped:    {raw!r} — supply a 4-digit Gregorian year "
                    f"or the Hebrew directly."
                )
            continue
        elif _DATE_PATTERN.search(raw):
            # Full date string (e.g. "August 14 2026", "2026-08-14", "14/8/2026").
            try:
                spaced, condensed, label = gregorian_to_hebrew_tokens(raw)
                if show_translations:
                    print(f"  Detected Date: {raw!r} \u2192 {condensed!r}  [{spaced}]")
                result.words.append(condensed)
                result.labels.append(label)
                result.origins[condensed] = label
            except Exception as exc:
                print(f"  Skipped:    {raw!r} \u2014 date parse failed: {exc}")
            continue  # skip synonym expansion for date tokens
        elif translate_english:
            # English (or other non-Hebrew) — translate
            hebrew = _translate_to_hebrew(raw, cache)
            label = f"{raw} → {hebrew}"
            if show_translations:
                print(f"  Translated: {raw!r} → {hebrew!r}")
            result.words.append(hebrew)
            result.labels.append(label)
            result.origins[hebrew] = label

            if expand:
                synonyms = _fetch_synonyms(raw, hebrew, expand_count, cache, method=expand_method)
                for syn in synonyms:
                    syn_label = f"{raw} (syn) → {syn}"
                    if show_translations:
                        print(f"  Synonym:    {raw!r} → {syn!r}")
                    result.words.append(syn)
                    result.labels.append(syn_label)
                    result.origins[syn] = syn_label
        else:
            # --translate not active and not a date/year/Hebrew — pass through as-is
            result.words.append(raw)
            result.labels.append(raw)
            result.origins[raw] = raw

    return result
