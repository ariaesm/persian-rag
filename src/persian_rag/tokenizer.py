"""Tokenization, light stemming and the analysis chain used by retrievers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from persian_rag.normalizer import ZWNJ, Normalizer
from persian_rag.stopwords import DEFAULT_STOPWORDS

# A token is a run of word characters, optionally joined by ZWNJ ("می‌روم", "کتاب‌ها").
_TOKEN = re.compile(r"\w+(?:\u200c\w+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split ``text`` into word tokens, keeping ZWNJ-joined words together.

    Punctuation, whitespace and symbols are dropped; Latin words are lower-cased.

    >>> tokenize("کتاب‌های من، Python!")
    ['کتاب‌های', 'من', 'python']
    """
    return [t.lower() for t in _TOKEN.findall(text)]


def count_tokens(text: str) -> int:
    """Return the number of word tokens in ``text``."""
    return sum(1 for _ in _TOKEN.finditer(text))


class PersianStemmer:
    """A light, conservative suffix/prefix stripper for Persian.

    Only affixes that are unambiguous are removed: ZWNJ-attached ones always, and a
    few common suffixes written without ZWNJ (``ها``, ``های``, ``ترین``) only when a
    stem of at least three letters remains. That keeps words like "تنها" and "بهترین"
    intact while folding "کتابها" and "کتاب‌ها" onto "کتاب".
    """

    _ZWNJ_SUFFIXES: tuple[str, ...] = tuple(
        ZWNJ + s
        for s in (
            "هایشان هایتان هایمان هایی هایم هایت هایش های ها ترین تر اند ایم اید ام اش ات ای ی"
        ).split()
    )
    _BARE_SUFFIXES: tuple[str, ...] = ("هایی", "های", "ها", "ترین")
    _PREFIXES: tuple[str, ...] = ("نمی" + ZWNJ, "می" + ZWNJ)

    def __init__(self, min_stem: int = 2) -> None:
        """Create a stemmer.

        Args:
            min_stem: Minimum number of characters a ZWNJ-stripped stem must keep.

        """
        self.min_stem = min_stem
        self._zwnj = tuple(sorted(set(self._ZWNJ_SUFFIXES), key=len, reverse=True))

    def stem(self, word: str) -> str:
        """Return the stem of a single normalized token."""
        for prefix in self._PREFIXES:
            if word.startswith(prefix) and len(word) - len(prefix) >= self.min_stem:
                word = word[len(prefix) :]
                break
        for suffix in self._zwnj:
            if word.endswith(suffix) and len(word) - len(suffix) >= self.min_stem:
                return word[: -len(suffix)]
        for suffix in self._BARE_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    def __call__(self, word: str) -> str:
        """Alias for :meth:`stem`."""
        return self.stem(word)


@dataclass
class Analyzer:
    """Text → terms pipeline: normalize, tokenize, drop stopwords, stem.

    Each stage can be switched off, which is how the evaluation measures the impact of
    Persian-specific processing.
    """

    normalizer: Callable[[str], str] | None = field(default_factory=Normalizer)
    stopwords: Iterable[str] | None = DEFAULT_STOPWORDS
    stemmer: Callable[[str], str] | None = field(default_factory=PersianStemmer)

    def __post_init__(self) -> None:
        """Freeze the stopword collection for fast membership tests."""
        self._stop = frozenset(self.stopwords or ())

    @classmethod
    def raw(cls) -> Analyzer:
        """Return an analyzer that only tokenizes (no normalization or stemming)."""
        return cls(normalizer=None, stopwords=DEFAULT_STOPWORDS, stemmer=None)

    def __call__(self, text: str) -> list[str]:
        """Analyze ``text`` into a list of index terms."""
        if self.normalizer is not None:
            text = self.normalizer(text)
        terms = [t for t in tokenize(text) if t not in self._stop]
        if self.stemmer is not None:
            terms = [self.stemmer(t) for t in terms]
        return terms
