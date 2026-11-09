"""A compact, hand-written list of Persian stopwords.

The list is intentionally small: high-frequency function words, auxiliaries and
question words that carry little topical signal for retrieval. All entries are already
in normalized form (Persian ی/ک, ZWNJ where standard).
"""

from __future__ import annotations

PERSIAN_STOPWORDS: frozenset[str] = frozenset(
    """
    و در به از که این آن با را برای تا بر یا اما هم نیز پس اگر چون چه ولی
    است هست نیست بود بودند باشد باشند شد شده شدند شود شوند می‌شود می‌شوند
    کرد کردند کرده کند کنند می‌کند می‌کنند کردن دارد دارند داشت داشتند
    خواهد خواهند توان می‌توان باید
    یک هر همه چند دیگر بسیار خیلی فقط حتی همین همان چنین آنها آن‌ها این‌ها ها
    من تو او ما شما ایشان خود خودش وی
    روی زیر بین میان پیش پس بعد قبل درباره مانند مثل طی نزد سوی
    ای آیا چرا چگونه چطور کجا کجاست کی کدام چیست کیست چیه چی
    را هم نه بله
    """.split()
)

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    "a an and are as at be by for from in is it of on or that the to was what with".split()
)

DEFAULT_STOPWORDS: frozenset[str] = PERSIAN_STOPWORDS | ENGLISH_STOPWORDS
