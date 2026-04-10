"""
expander.py – Static Biblical Hebrew lexicon for ELS synonym expansion.

Provides deterministic, network-free synonym lookup keyed by English concept
names or by Hebrew consonant strings directly.

The lexicon focuses on two research themes:
  • "Leader / Ruler" (נשיא, מלך, שר, ראש, מושל, אדון, נגיד, …)
  • "Messianic / Salvation" (משיח, ישועה, גואל, מלך, דוד, …)

plus several other high-frequency Biblical concept clusters.

Public API
----------
get_synonyms(word, normalise_fn=None) → list[str]
    Accept either an English key (case-insensitive) or a Hebrew consonant
    string.  Returns a deduplicated list of related Hebrew consonant strings
    from the static lexicon, excluding the input word itself.

    ``normalise_fn`` should be ``translator._normalise_hebrew`` or equivalent
    if you want the Hebrew look-up normalised before matching; defaults to
    identity transform.

ENGLISH_TO_HEBREW : dict[str, list[str]]
    Raw English→Hebrew mapping; importable for inspection / extension.

HEBREW_TO_RELATED : dict[str, list[str]]
    Raw Hebrew→related-Hebrew mapping (consonants only, non-final forms).
"""

from __future__ import annotations

from typing import Callable, Sequence

# ── Static lexicon ─────────────────────────────────────────────────────────────
#
# Rules:
#  - Hebrew values use non-final consonant forms only (ם→מ, ך→כ, ן→נ, ף→פ, ץ→צ)
#    to match the compact encoding used by the corpus builder.
#  - Every list is ordered from most-specific / most-significant to broader synonyms.
#  - Roots with multiple meanings appear in multiple theme clusters intentionally.
#

# English concept → list of Hebrew equivalents (consonants, non-final forms)
ENGLISH_TO_HEBREW: dict[str, list[str]] = {
    # ── Leader / Ruler theme ────────────────────────────────────────────────
    "president":  ["נשיא", "ראש", "שר"],
    "leader":     ["נשיא", "ראש", "מנהיג", "שר", "נגיד"],
    "ruler":      ["מושל", "מלך", "שלט", "נגיד"],
    "king":       ["מלך", "אדניר", "שלמה"],
    "prince":     ["נשיא", "שר", "נגיד"],
    "chief":      ["ראש", "נשיא", "שר"],
    "lord":       ["אדון", "אדני", "בעל"],
    "governor":   ["פחה", "מושל", "שר"],
    "judge":      ["שפט", "דין"],
    "captain":    ["שר", "נגיד", "קצן"],
    "commander":  ["שר", "נגיד", "קצן"],

    # ── Messianic / Salvation theme ─────────────────────────────────────────
    "messiah":    ["משיח", "ישועה", "גואל", "נצר"],
    "savior":     ["ישועה", "גואל", "פדה", "נצל"],
    "salvation":  ["ישועה", "ישע", "פדות", "גאלה"],
    "redeemer":   ["גואל", "פדה", "גאל"],
    "anointed":   ["משיח", "שמן"],
    "peace":      ["שלומ", "שלוה"],
    "covenant":   ["ברית", "חוק"],
    "holy":       ["קדש", "קדוש"],
    "glory":      ["כבוד", "הוד", "תפארה"],

    # ── Named / transliterated figures ─────────────────────────────────────
    "trump":      ["טראמפ"],
    "donald":     ["דונלד"],
    "cyrus":      ["כורש"],
    "david":      ["דוד"],
    "solomon":    ["שלמה"],
    "moses":      ["משה"],
    "elijah":     ["אליה"],
    "abraham":    ["אברהמ"],             # final mem → מ
    "israel":     ["ישראל"],
    "jacob":      ["יעקב"],
    "jerusalem":  ["ירושלמ"],            # Jerusalem (non-final mem)
    "zion":       ["ציון"],
    "daniel":     ["דניאל"],
    "ezekiel":    ["יחזקאל"],
    "isaiah":     ["ישעיה"],
    "jeremiah":   ["ירמיה"],

    # ── Core theological concepts ───────────────────────────────────────────
    "torah":      ["תורה"],
    "law":        ["תורה", "חוק", "מצוה"],
    "god":        ["אלהימ", "אל", "שדי"],   # Elohim (final mem → מ)
    "lord":       ["יהוה", "אדון", "אדני"],
    "temple":     ["מקדש", "בית", "היכל"],
    "sacrifice":  ["קרבן", "זבח", "עולה"],
    "prayer":     ["תפלה", "פגר"],
    "sin":        ["חטא", "פשע", "עוון"],
    "forgiveness":["סלח", "כפר", "מחל"],
    "repentance": ["שוב", "תשובה"],
    "faith":      ["אמונה", "בטח"],
    "love":       ["אהבה", "חסד"],
    "blessing":   ["ברכה", "טוב"],
    "curse":      ["ארר", "קלל"],
    "prophecy":   ["נבואה", "נבא"],
    "war":        ["מלחמה", "קרב", "חרב"],
    "nation":     ["גוי", "עמ", "לאמ"],    # non-final mem
    "people":     ["עמ", "לאמ", "עדה"],
    "land":       ["ארצ", "אדמה"],         # non-final tsade
    "exile":      ["גלה", "גולה"],
    "return":     ["שוב", "שיבה"],
    "end":        ["קצ", "אחרית", "סוף"],  # non-final tsade
    "days":       ["ימימ", "יומ"],          # non-final mem
    "light":      ["אור", "נר"],
    "darkness":   ["חשך", "עלטה"],
    "fire":       ["אש", "אור"],
    "water":      ["מימ", "נהר"],          # non-final mem
    "wisdom":     ["חכמה", "בינה", "דעת"],
    "truth":      ["אמת", "צדק"],
    "justice":    ["משפט", "צדק", "דין"],
    "righteousness": ["צדקה", "צדק"],
    "spirit":     ["רוח", "נשמה"],
    "heart":      ["לב", "לבב"],
    "soul":       ["נפש", "נשמה", "רוח"],
    "life":       ["חיימ", "חיה"],          # non-final mem
    "death":      ["מות", "מת"],
    "blood":      ["דמ"],                  # non-final mem
    "hand":       ["יד", "כפ"],            # non-final pe
    "sword":      ["חרב", "חרבה"],
    "book":       ["ספר", "כתב"],
}

# Hebrew consonant string → related Hebrew consonant strings
# Keys and values must use non-final consonant forms (matching corpus encoding).
HEBREW_TO_RELATED: dict[str, list[str]] = {
    # Leader cluster
    "נשיא":  ["ראש", "שר", "מלך", "מושל", "נגיד"],
    "מלך":   ["נשיא", "שר", "נגיד", "מושל"],
    "שר":    ["נשיא", "ראש", "נגיד", "מלך"],
    "ראש":   ["נשיא", "שר", "מנהיג"],
    "נגיד":  ["מלך", "נשיא", "שר", "מושל"],
    "מושל":  ["מלך", "נשיא", "שלט"],
    "אדון":  ["מלך", "אדני", "יהוה"],

    # Messianic cluster
    "משיח":  ["ישועה", "גואל", "נצר", "דוד"],
    "ישועה": ["משיח", "גואל", "ישע", "פדות"],
    "ישע":   ["ישועה", "גואל", "פדה"],
    "גואל":  ["ישועה", "משיח", "פדה", "גאל"],
    "פדה":   ["גואל", "גאל", "ישועה"],
    "נצר":   ["משיח", "דוד", "ישועה"],

    # Peace / Covenant cluster
    "שלומ":  ["שלוה", "ברית"],
    "שלוה":  ["שלומ"],
    "ברית":  ["חוק", "אמנה", "שלומ"],
    "חוק":   ["ברית", "מצוה", "תורה"],

    # Divine / Holy cluster
    "קדש":   ["קדוש", "כבוד", "הוד"],
    "קדוש":  ["קדש", "כבוד"],
    "כבוד":  ["הוד", "תפארה", "קדוש"],
    "יהוה":  ["אלהימ", "אדון", "אל", "שדי"],
    "אלהימ": ["יהוה", "אל", "שדי", "אדון"],

    # Key figures
    "דוד":   ["שלמה", "נשיא", "משיח", "מלך"],
    "שלמה":  ["דוד", "מלך"],
    "משה":   ["אהרנ", "נביא", "לוי"],      # non-final nun/mem
    "אברהמ": ["ישראל", "יצחק", "יעקב"],
    "ישראל": ["יעקב", "אברהמ"],
    "יעקב":  ["ישראל", "אברהמ", "יצחק"],

    # Places
    "ירושלמ":["ציון", "בית", "מקדש"],
    "ציון":  ["ירושלמ", "מקדש", "הר"],

    # Prophetic / Eschatological
    "נבואה": ["נבא", "חזיון"],
    "קצ":    ["אחרית", "סוף"],             # non-final tsade
    "אחרית": ["קצ", "סוף"],
    "גלה":   ["גולה", "שוב"],
    "שוב":   ["גלה", "תשובה"],

    # Spiritual
    "תורה":  ["מצוה", "חוק", "ברית"],
    "מצוה":  ["תורה", "חוק"],
    "אמת":   ["צדק", "אמונה"],
    "צדק":   ["משפט", "צדקה", "אמת"],
    "צדקה":  ["צדק", "חסד"],
    "חסד":   ["אהבה", "אמת"],
    "אהבה":  ["חסד"],
    "אמונה": ["אמת", "בטח"],
    "רוח":   ["נשמה", "נפש"],
    "נפש":   ["רוח", "נשמה", "לב"],
    "חיימ":  ["נפש", "נשמה"],
    "מות":   ["קברה"],
    "חכמה":  ["בינה", "דעת"],
    "בינה":  ["חכמה", "דעת"],
    "דעת":   ["חכמה", "בינה"],
    "ברכה":  ["טוב", "שלומ"],
    "טוב":   ["ברכה"],
    "אור":   ["נר", "אש"],
    "חשך":   ["עלטה"],
    "ספר":   ["כתב", "תורה"],
    "שמימ":  ["ארצ", "עננ"],              # non-final mem/nun
    "ארצ":   ["אדמה", "שמימ"],
}


def get_synonyms(
    word: str,
    normalise_fn: Callable[[str], str] | None = None,
    count: int | None = None,
) -> list[str]:
    """
    Return related Hebrew consonant strings for *word* from the static lexicon.

    Parameters
    ----------
    word :
        Either an English concept name (case-insensitive, e.g. ``"President"``)
        or a Hebrew consonant string (e.g. ``"נשיא"``).
    normalise_fn :
        Optional function to normalise a Hebrew string before dictionary lookup
        (e.g., ``translator._normalise_hebrew``).  Pass ``None`` to skip.
    count :
        Cap the number of synonyms returned.  ``None`` = return all.

    Returns
    -------
    list[str]
        Deduplicated related Hebrew strings, excluding *word* itself.
        Empty list if *word* is not in the lexicon.
    """
    # Determine if the input is Hebrew (contains a character in U+05D0–U+05EA)
    _HEB_RE_CHECK = set(range(0x05D0, 0x05EB))
    is_hebrew = any(ord(ch) in _HEB_RE_CHECK for ch in word)

    if is_hebrew:
        # Normalise first if a function is provided
        key = normalise_fn(word) if normalise_fn else word
        related = HEBREW_TO_RELATED.get(key, [])
    else:
        key = word.strip().lower()
        # English lookup: gather all Hebrew equivalents beyond the first one
        hebrew_list = ENGLISH_TO_HEBREW.get(key, [])
        if not hebrew_list:
            return []
        # Primary = first entry; "synonyms" = remaining entries
        related = hebrew_list[1:]

    # Exclude the input word itself (normalised, if applicable)
    exclusion = (normalise_fn(word) if normalise_fn else word) if is_hebrew else None
    result = [h for h in related if h != exclusion]

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for h in result:
        if h not in seen:
            seen.add(h)
            deduped.append(h)

    return deduped[:count] if count is not None else deduped


def get_all_hebrew(
    word: str,
    normalise_fn: Callable[[str], str] | None = None,
) -> list[str]:
    """
    Return *all* Hebrew strings associated with *word*, including the primary
    translation (if English) as the first element.

    For Hebrew inputs, returns ``[word] + get_synonyms(word, ...)``.
    For English inputs, returns the full ``ENGLISH_TO_HEBREW`` list.

    Useful when you want to fold the primary form and all synonyms into one
    flat list without a separate translation step.
    """
    _HEB_RE_CHECK = set(range(0x05D0, 0x05EB))
    is_hebrew = any(ord(ch) in _HEB_RE_CHECK for ch in word)

    if is_hebrew:
        key = normalise_fn(word) if normalise_fn else word
        syns = get_synonyms(word, normalise_fn)
        result = [key] + syns
    else:
        key = word.strip().lower()
        result = ENGLISH_TO_HEBREW.get(key, [])

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for h in result:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
    return deduped
