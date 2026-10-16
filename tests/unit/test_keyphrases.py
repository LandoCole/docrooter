from docrooter.keyphrases import STOPWORDS, extract_keyphrases


def test_keyphrases_prefer_distinctive_terms():
    sec = (
        "The revolving credit facility matures in 2027. "
        "The revolving credit facility has covenants."
    )
    others = ["The company reported revenue growth.", "The company hired staff."]
    kp = extract_keyphrases(sec, [sec, *others], k=3)
    assert kp[0] == "revolving credit facility" and "the company" not in kp


def test_candidates_never_start_or_end_with_stopword():
    text = "The board of directors approved the plan for the merger."
    for phrase in extract_keyphrases(text, [text], k=20):
        words = phrase.split()
        assert words[0] not in STOPWORDS and words[-1] not in STOPWORDS


def test_substring_phrases_dropped_and_k_respected():
    sec = "the interest rate swap, an interest rate swap, and the interest rate swap"
    kp = extract_keyphrases(sec, [sec, "other text entirely"], k=5)
    assert kp == ["interest rate swap"]
    assert len(extract_keyphrases("alpha beta gamma delta epsilon", ["x"], k=2)) == 2


def test_empty_text_and_zero_k():
    assert extract_keyphrases("", ["a b"]) == []
    assert extract_keyphrases("alpha", ["alpha"], k=0) == []


def test_word_boundary_substring_not_dropped():
    sec = "category category category and art"
    kp = extract_keyphrases(sec, [sec, "zzz"], k=5)
    assert "art" in kp
