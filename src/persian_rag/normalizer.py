"""Persian text normalization, written from scratch.

Persian text found in the wild is remarkably inconsistent: the same word can be typed
with Arabic or Persian code points, with or without diacritics, with a space, a
zero-width non-joiner (ZWNJ, "نیم‌فاصله") or nothing between a stem and its affixes,
and with three different digit systems. To a byte-level search engine these are all
different strings. :class:`Normalizer` maps them onto one canonical form.

Example:
    >>> from persian_rag.normalizer import normalize
    >>> normalize("كتاب هاي  تاريخي")
    'کتاب‌های تاریخی'

"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

DigitMode = Literal["fa", "en", "keep"]

ZWNJ = "\u200c"

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"

# Letters of the Arabic script block that can appear in Persian words. Digits
# (U+0660-U+0669, U+06F0-U+06F9) and punctuation are deliberately excluded.
LETTER_CHARS = "\u0621-\u063a\u0641-\u064a\u0671-\u06d3\u06d5\u06fa-\u06fc"
LETTER = f"[{LETTER_CHARS}]"

# Arabic code points that have a dedicated Persian equivalent.
_CHAR_MAP: dict[str, str] = {
    "\u064a": "\u06cc",  # ي ARABIC YEH        -> ی FARSI YEH
    "\u0649": "\u06cc",  # ى ALEF MAKSURA      -> ی
    "\u0643": "\u06a9",  # ك ARABIC KAF        -> ک KEHEH
    "\u0629": "\u0647",  # ة TEH MARBUTA       -> ه
    "\u0623": "\u0627",  # أ ALEF WITH HAMZA ABOVE -> ا
    "\u0625": "\u0627",  # إ ALEF WITH HAMZA BELOW -> ا
    "\u0671": "\u0627",  # ٱ ALEF WASLA        -> ا
    "\u06d5": "\u0647",  # ە AE                -> ه
}

# ۀ (heh with yeh above) and ه + combining hamza both spell the ezafe "ه‌ی".
_HEH_YEH = re.compile("\u06c0|\u0647\u0654")

# Harakat, tanwin, shadda, sukun, small high marks and superscript alef.
_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TATWEEL = re.compile("\u0640+")
# Directional marks and other invisible formatting characters (ZWNJ is handled apart).
_INVISIBLE = re.compile("[\u200b\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")

_SPACES = re.compile(r"[ \t\f\v\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")

# Verbal prefixes "می" / "نمی" written as a separate word: "می روم" -> "می‌روم".
_PREFIX = re.compile(rf"(?<![{LETTER_CHARS}\u200c])(ن?می)[ \u200c]+(?={LETTER})")
# Plural / comparative / superlative suffixes written as a separate word.
_SUFFIX = re.compile(
    rf"({LETTER})[ \u200c]+"
    r"(هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان|های|ها|ترین|تر)"
    rf"(?!{LETTER})"
)
# Ezafe and copula/pronominal endings after a final heh: "خانه ی" / "رفته اند".
_HEH_SUFFIX = re.compile(rf"(ه)[ \u200c]+(ی|ام|ای|ایم|اید|اند|اش)(?!{LETTER})")
_ZWNJ_RUNS = re.compile("\u200c{2,}")
_ZWNJ_NEAR_SPACE = re.compile(r"\u200c*(\s)\u200c*")
_ZWNJ_EDGES = re.compile(rf"(?<!{LETTER})\u200c|\u200c(?!{LETTER})")

_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([،؛؟!:.»)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([«(\[])[ \t]+")
_LATIN_PUNCT_AFTER_PERSIAN = re.compile(rf"(?<=[{LETTER_CHARS}»])[ \t]*([,;?])")
_PUNCT_NEEDS_SPACE = re.compile(rf"([،؛؟!:.])(?={LETTER}|«)")
_STRAIGHT_QUOTES = re.compile(rf'"([^"\n]*{LETTER}[^"\n]*)"')
_LATIN_TO_PERSIAN_PUNCT = {",": "،", ";": "؛", "?": "؟"}


@dataclass(frozen=True, slots=True)
class NormalizerConfig:
    """Switches for each normalization step.

    Attributes:
        unicode_nfkc: Apply NFKC first (folds Arabic presentation forms found in PDFs).
        unify_characters: Map Arabic ي/ك/ة/ۀ (and friends) to their Persian forms.
        remove_diacritics: Strip harakat such as fatha, kasra, damma, tanwin and shadda.
        remove_tatweel: Strip the kashida/tatweel elongation character (ـ).
        digits: Convert all digits to Persian (``"fa"``), Latin (``"en"``) or ``"keep"``.
        fix_zwnj: Insert ZWNJ around common affixes and clean up stray ZWNJs.
        normalize_punctuation: Use Persian punctuation and fix spacing around it.
        collapse_whitespace: Collapse runs of spaces and blank lines.

    """

    unicode_nfkc: bool = True
    unify_characters: bool = True
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    digits: DigitMode = "en"
    fix_zwnj: bool = True
    normalize_punctuation: bool = True
    collapse_whitespace: bool = True


def _digit_table(mode: DigitMode) -> dict[int, str]:
    if mode == "keep":
        return {}
    if mode == "fa":
        sources, target = ARABIC_DIGITS + LATIN_DIGITS, PERSIAN_DIGITS * 2
        table = str.maketrans(sources, target)
        table[ord("٫")] = "٫"
        return table
    if mode == "en":
        sources, target = ARABIC_DIGITS + PERSIAN_DIGITS, LATIN_DIGITS * 2
        table = str.maketrans(sources, target)
        table[ord("٫")] = "."  # Arabic decimal separator
        return table
    raise ValueError(f"unknown digit mode: {mode!r} (expected 'fa', 'en' or 'keep')")


class Normalizer:
    """Configurable, deterministic Persian normalizer.

    Instances are cheap and stateless after construction; reuse one across calls.
    """

    def __init__(self, config: NormalizerConfig | None = None, **overrides: object) -> None:
        """Create a normalizer from a config and/or keyword overrides.

        Args:
            config: Base configuration. Defaults to :class:`NormalizerConfig`.
            **overrides: Field overrides, e.g. ``Normalizer(digits="fa")``.

        """
        base = config or NormalizerConfig()
        if overrides:
            fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
            unknown = set(overrides) - set(fields)
            if unknown:
                raise TypeError(f"unknown normalizer option(s): {sorted(unknown)}")
            fields.update(overrides)
            base = NormalizerConfig(**fields)  # type: ignore[arg-type]
        self.config = base
        self._char_table = str.maketrans(_CHAR_MAP)
        self._digits = _digit_table(base.digits)

    def __call__(self, text: str) -> str:
        """Alias for :meth:`normalize`."""
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """Return the normalized form of ``text``."""
        cfg = self.config
        if cfg.unicode_nfkc:
            text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _INVISIBLE.sub("", text)
        if cfg.unify_characters:
            text = _HEH_YEH.sub("\u0647\u200c\u06cc", text)
            text = text.translate(self._char_table)
        if cfg.remove_diacritics:
            text = _DIACRITICS.sub("", text)
        if cfg.remove_tatweel:
            text = _TATWEEL.sub("", text)
        if self._digits:
            text = text.translate(self._digits)
        if cfg.normalize_punctuation:
            text = self._punctuation(text)
        if cfg.collapse_whitespace:
            text = _SPACES.sub(" ", text)
        if cfg.fix_zwnj:
            text = self._zwnj(text)
        if cfg.collapse_whitespace:
            text = _SPACES.sub(" ", text)
            text = "\n".join(line.strip() for line in text.split("\n"))
            text = _MANY_NEWLINES.sub("\n\n", text).strip()
        return text

    @staticmethod
    def _punctuation(text: str) -> str:
        text = _STRAIGHT_QUOTES.sub(r"«\1»", text)
        text = _LATIN_PUNCT_AFTER_PERSIAN.sub(lambda m: _LATIN_TO_PERSIAN_PUNCT[m.group(1)], text)
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _SPACE_AFTER_OPEN.sub(r"\1", text)
        return _PUNCT_NEEDS_SPACE.sub(r"\1 ", text)

    @staticmethod
    def _zwnj(text: str) -> str:
        text = _PREFIX.sub(r"\1" + ZWNJ, text)
        text = _SUFFIX.sub(r"\1" + ZWNJ + r"\2", text)
        text = _HEH_SUFFIX.sub(r"\1" + ZWNJ + r"\2", text)
        text = _ZWNJ_RUNS.sub(ZWNJ, text)
        text = _ZWNJ_NEAR_SPACE.sub(r"\1", text)
        return _ZWNJ_EDGES.sub("", text)


_DEFAULT = Normalizer()


def normalize(text: str, **options: object) -> str:
    """Normalize ``text`` with the default configuration (or keyword overrides).

    Args:
        text: Raw Persian (or mixed-language) text.
        **options: Any :class:`NormalizerConfig` field, e.g. ``digits="fa"``.

    Returns:
        The normalized text.

    """
    if not options:
        return _DEFAULT.normalize(text)
    return Normalizer(**options).normalize(text)
