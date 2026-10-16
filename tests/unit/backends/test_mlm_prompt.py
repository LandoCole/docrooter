from docrooter.backends.mlm_choice import LETTERS, build_choice_prompt


def test_choice_prompt_format():
    prompt = build_choice_prompt(
        "Deferred taxes rose.",
        ["Tax: income taxes", "Litigation: lawsuits"],
        instruction="Pick the best label.",
        answer="[MASK]",
    )
    assert prompt == (
        "Pick the best label.\n"
        "Text: Deferred taxes rose.\n"
        "Options:\n"
        "A. Tax: income taxes\n"
        "B. Litigation: lawsuits\n"
        "Answer: [MASK]"
    )


def test_choice_prompt_empty_answer_has_no_trailing_space():
    prompt = build_choice_prompt("t", ["x", "y"], instruction="I", answer="")
    assert prompt.endswith("B. y\nAnswer:")


def test_custom_markers_and_letters():
    assert LETTERS[0] == "A" and LETTERS[-1] == "Z" and len(LETTERS) == 26
    prompt = build_choice_prompt("t", ["x", "y"], instruction="I", answer="?", markers=("1", "2"))
    assert "Options:\n1. x\n2. y\nAnswer: ?" in prompt
