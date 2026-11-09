from persian_rag.normalizer import ZWNJ
from persian_rag.stopwords import PERSIAN_STOPWORDS
from persian_rag.tokenizer import Analyzer, PersianStemmer, count_tokens, tokenize


def test_tokenize_keeps_zwnj_words_and_drops_punctuation() -> None:
    text = f"کتاب{ZWNJ}های من، «پایتون» و Python!"
    assert tokenize(text) == [f"کتاب{ZWNJ}های", "من", "پایتون", "و", "python"]


def test_tokenize_numbers_and_mixed_scripts() -> None:
    assert tokenize("سال 1403 (2024)") == ["سال", "1403", "2024"]
    assert count_tokens("یک دو سه.") == 3


def test_stopwords_are_normalized_forms() -> None:
    from persian_rag.normalizer import normalize

    for word in PERSIAN_STOPWORDS:
        assert normalize(word) == word, word


def test_stemmer_strips_zwnj_suffixes() -> None:
    s = PersianStemmer()
    assert s(f"کتاب{ZWNJ}ها") == "کتاب"
    assert s(f"کتاب{ZWNJ}های") == "کتاب"
    assert s(f"بزرگ{ZWNJ}ترین") == "بزرگ"
    assert s(f"بزرگ{ZWNJ}تر") == "بزرگ"
    assert s(f"رفته{ZWNJ}اند") == "رفته"
    assert s(f"خانه{ZWNJ}ی") == "خانه"


def test_stemmer_strips_verbal_prefix() -> None:
    s = PersianStemmer()
    assert s(f"می{ZWNJ}کارند") == "کارند"
    assert s(f"نمی{ZWNJ}دانم") == "دانم"


def test_stemmer_is_conservative_without_zwnj() -> None:
    s = PersianStemmer()
    assert s("کتابها") == "کتاب"
    assert s("تنها") == "تنها"  # "alone", not a plural
    assert s("بهترین") == "بهترین"  # stem would be too short
    assert s("دختر") == "دختر"  # bare "تر" is never stripped


def test_analyzer_full_chain_unifies_variants() -> None:
    analyze = Analyzer()
    a = analyze("كتابهاي تاريخي")
    b = analyze(f"کتاب{ZWNJ}های تاریخی")
    c = analyze("کتاب های تاریخی")
    assert a == b == c == ["کتاب", "تاریخی"]


def test_analyzer_removes_stopwords() -> None:
    assert Analyzer()("زعفران در کجا کشت می شود؟") == ["زعفران", "کشت"]


def test_raw_analyzer_does_not_normalize() -> None:
    raw = Analyzer.raw()
    assert raw("كتاب") != raw("کتاب")
