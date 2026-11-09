import pytest

from persian_rag.schema import Chunk


def make_chunks(texts: dict[str, str]) -> list[Chunk]:
    return [Chunk(id=f"{d}#0", doc_id=d, index=0, text=t) for d, t in texts.items()]


@pytest.fixture
def chunks() -> list[Chunk]:
    # Deliberately noisy orthography, as found in real Persian web text.
    return make_chunks(
        {
            "saffron": "زعفران در خراسان کشت مي شود و گل هاي آن در پاييز برداشت مي شوند.",
            "metro": "متروی تهران چند خط دارد و روزانه مسافران زیادی را جابه‌جا می‌کند.",
            "hafez": "حافظ شاعر بزرگ شيراز در قرن هشتم هجري زندگي مي‌كرد.",
            "python": "پایتون یک زبان برنامه‌نویسی خوانا است که در سال ۱۹۹۱ منتشر شد.",
        }
    )
