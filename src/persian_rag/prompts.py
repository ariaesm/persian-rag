"""Persian prompts that keep the model grounded in retrieved passages."""

from __future__ import annotations

from collections.abc import Sequence

from persian_rag.schema import SearchResult

IDK = "نمی‌دانم"

SYSTEM_PROMPT = f"""تو یک دستیار پرسش و پاسخ فارسی هستی که فقط بر اساس «منابع» داده‌شده پاسخ می‌دهی.

قوانین:
۱. فقط از اطلاعات موجود در منابع استفاده کن و هیچ چیزی از دانش خودت اضافه نکن.
۲. پس از هر جمله یا ادعا، شماره‌ی منبع پشتیبان را داخل کروشه بیاور، مثل [1] یا [2][3].
۳. اگر پاسخ در منابع نیامده یا منابع کافی نیستند، فقط و دقیقاً بنویس: {IDK}
۴. پاسخ را به زبان فارسی روان، دقیق و کوتاه بنویس."""

USER_TEMPLATE = """منابع:
{context}

پرسش: {question}

پاسخ (با ارجاع به شماره‌ی منابع):"""


def format_context(results: Sequence[SearchResult], max_chars: int = 1500) -> str:
    """Render retrieved chunks as numbered sources: ``[1] (منبع: file) text``."""
    blocks = []
    for i, hit in enumerate(results, start=1):
        text = hit.chunk.text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        blocks.append(f"[{i}] (منبع: {hit.chunk.doc_id})\n{text}")
    return "\n\n".join(blocks)


def build_messages(question: str, results: Sequence[SearchResult]) -> list[dict[str, str]]:
    """Return chat messages for a grounded answer to ``question``."""
    user = USER_TEMPLATE.format(context=format_context(results), question=question.strip())
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
