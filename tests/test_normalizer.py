import pytest

from persian_rag.normalizer import ZWNJ, Normalizer, NormalizerConfig, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ايران", "ایران"),  # Arabic yeh
        ("كتاب", "کتاب"),  # Arabic kaf
        ("مدرسة", "مدرسه"),  # teh marbuta
        ("علي", "علی"),
        ("مصطفى", "مصطفی"),  # alef maksura
        ("أحمد", "احمد"),
    ],
)
def test_arabic_letters_become_persian(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_heh_with_yeh_above_becomes_ezafe() -> None:
    assert normalize("خانۀ ما") == f"خانه{ZWNJ}ی ما"
    assert normalize("خانهٔ ما") == f"خانه{ZWNJ}ی ما"


def test_diacritics_are_removed() -> None:
    assert normalize("عِلْمٌ") == "علم"
    assert normalize("مُحَمَّد") == "محمد"
    assert normalize("دانشگاهً") == "دانشگاه"


def test_tatweel_is_removed() -> None:
    assert normalize("سلامـــــت") == "سلامت"
    assert normalize("بـزرگ") == "بزرگ"


def test_alef_madda_survives_nfkc_and_diacritic_removal() -> None:
    # Decomposed alef + madda must recompose to آ, not lose the madda.
    assert normalize("آب") == "آب"


def test_digits_default_to_latin() -> None:
    assert normalize("سال ۱۴۰۳ و ٢٠٢٤") == "سال 1403 و 2024"


def test_digits_can_be_persian() -> None:
    assert normalize("سال 1403 و ٢٠٢٤", digits="fa") == "سال ۱۴۰۳ و ۲۰۲۴"


def test_digits_can_be_kept() -> None:
    assert normalize("۱۲ و 34", digits="keep") == "۱۲ و 34"


def test_arabic_decimal_separator_in_latin_mode() -> None:
    assert normalize("۳٫۵ کیلو") == "3.5 کیلو"


def test_invalid_digit_mode() -> None:
    with pytest.raises(ValueError, match="digit mode"):
        Normalizer(digits="roman")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("می روم", f"می{ZWNJ}روم"),
        ("نمی دانم", f"نمی{ZWNJ}دانم"),
        ("کتاب ها", f"کتاب{ZWNJ}ها"),
        ("کتاب های قدیمی", f"کتاب{ZWNJ}های قدیمی"),
        ("بزرگ تر", f"بزرگ{ZWNJ}تر"),
        ("بزرگ ترین شهر", f"بزرگ{ZWNJ}ترین شهر"),
        ("رفته اند", f"رفته{ZWNJ}اند"),
        ("خسته ام", f"خسته{ZWNJ}ام"),
    ],
)
def test_zwnj_inserted_around_affixes(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_zwnj_does_not_break_unrelated_words() -> None:
    # "میز" (table) and "تهران" must not be touched; "میوه ها" gets a suffix ZWNJ only.
    assert normalize("میز تهران") == "میز تهران"
    assert normalize("میوه ها") == f"میوه{ZWNJ}ها"
    assert normalize("دختر خوب") == "دختر خوب"


def test_stray_and_duplicate_zwnj_are_cleaned() -> None:
    assert normalize(f"کتاب{ZWNJ}{ZWNJ}ها {ZWNJ}و{ZWNJ}") == f"کتاب{ZWNJ}ها و"


def test_invisible_bidi_marks_removed() -> None:
    assert normalize("‏سلام‎") == "سلام"


def test_whitespace_collapsed_but_paragraphs_kept() -> None:
    raw = "  سلام   دنیا \t!\n\n\n\nخط   دوم  "
    assert normalize(raw) == "سلام دنیا!\n\nخط دوم"


def test_persian_punctuation() -> None:
    assert normalize("چرا?") == "چرا؟"
    assert normalize("سیب , پرتقال ; موز") == "سیب، پرتقال؛ موز"
    assert normalize('او گفت "سلام" و رفت') == "او گفت «سلام» و رفت"
    assert normalize("« سلام »") == "«سلام»"
    assert normalize("سلام،خوبی؟بله") == "سلام، خوبی؟ بله"


def test_english_punctuation_untouched() -> None:
    assert normalize("Python 3.11, released?") == "Python 3.11, released?"
    assert normalize("1,000,000") == "1,000,000"


def test_presentation_forms_from_pdfs_are_folded() -> None:
    # "سلام" encoded with Arabic presentation forms, as some PDF extractors emit.
    assert normalize("ﺳﻼﻡ") == "سلام"


def test_steps_can_be_disabled() -> None:
    n = Normalizer(NormalizerConfig(unify_characters=False, remove_diacritics=False))
    assert n("كِتاب") == "كِتاب"


def test_unknown_option_rejected() -> None:
    with pytest.raises(TypeError):
        Normalizer(spellcheck=True)


def test_idempotent_on_realistic_sentence() -> None:
    raw = "زبان‌هاي برنامه نويسي مثل پايتون در سال ۲۰۲۴ محبوب ترين بوده اند ."
    once = normalize(raw)
    assert once == normalize(once)
    assert (
        once == f"زبان{ZWNJ}های برنامه نویسی مثل پایتون در سال 2024 محبوب{ZWNJ}ترین بوده{ZWNJ}اند."
    )
