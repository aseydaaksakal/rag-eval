import pytest

from rag_eval.judges import LexicalJudge, LLMJudge, parse_score
from rag_eval.judges.lexical import overlap, split_sentences, tokenize

CONTEXT = ["Refunds are accepted within a 30 day window from the delivery date."]


def test_stopwords_are_dropped():
    assert tokenize("The refund is in the window") == ["refund", "window"]


def test_sentence_splitting():
    assert split_sentences("One. Two!\nThree?") == ["One.", "Two!", "Three?"]
    assert split_sentences("") == []


def test_overlap_ignores_function_words():
    assert overlap("the refund window", CONTEXT[0]) == pytest.approx(1.0)
    assert overlap("the shipping carrier", CONTEXT[0]) == pytest.approx(0.0)


def test_grounded_sentence_scores_one():
    judge = LexicalJudge()
    assert judge.groundedness("The refund window is 30 days.", CONTEXT) == 1.0


def test_ungrounded_sentence_scores_zero():
    judge = LexicalJudge()
    assert judge.groundedness("Express shipping costs nine euros.", CONTEXT) == 0.0


def test_partial_grounding_is_a_fraction():
    judge = LexicalJudge()
    answer = "The refund window is 30 days. Express shipping costs nine euros."
    assert judge.groundedness(answer, CONTEXT) == pytest.approx(0.5)


def test_unsupported_sentences_are_reported_for_debugging():
    judge = LexicalJudge()
    answer = "The refund window is 30 days. Express shipping costs nine euros."
    unsupported = judge.unsupported_sentences(answer, CONTEXT)
    assert unsupported == ["Express shipping costs nine euros."]


def test_support_is_best_of_passages_not_pooled():
    # Neither passage alone supports the claim, so neither should the pair.
    judge = LexicalJudge(threshold=0.9)
    assert judge.supported("alpha beta gamma", ["alpha beta", "gamma delta"]) < 0.9


def test_empty_answer_and_no_context():
    judge = LexicalJudge()
    assert judge.groundedness("", CONTEXT) == 0.0
    assert judge.groundedness("Anything.", []) == 0.0


def test_threshold_is_validated():
    with pytest.raises(ValueError):
        LexicalJudge(threshold=1.5)


def test_similarity_is_symmetric():
    judge = LexicalJudge()
    a, b = "refund window thirty days", "thirty day refund window"
    assert judge.similarity(a, b) == judge.similarity(b, a) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "reply, expected",
    [("0.8", 0.8), ("Score: 0.25", 0.25), ("1", 1.0), ("0", 0.0), ("75%", 0.75)],
)
def test_parse_score(reply, expected):
    assert parse_score(reply) == pytest.approx(expected)


def test_parse_score_rejects_prose():
    with pytest.raises(ValueError, match="no score found"):
        parse_score("I think it is mostly supported")


def test_llm_judge_caches_identical_prompts():
    calls = []

    def complete(prompt):
        calls.append(prompt)
        return "0.9"

    judge = LLMJudge(complete)
    judge.groundedness("An answer.", CONTEXT)
    judge.groundedness("An answer.", CONTEXT)
    assert len(calls) == 1


def test_llm_judge_can_swallow_errors():
    def broken(prompt):
        raise RuntimeError("provider down")

    assert LLMJudge(broken, on_error="zero").groundedness("a", CONTEXT) == 0.0
    with pytest.raises(RuntimeError):
        LLMJudge(broken).groundedness("a", CONTEXT)


def test_llm_judge_short_circuits_on_empty_input():
    judge = LLMJudge(lambda p: "1.0")
    assert judge.groundedness("   ", CONTEXT) == 0.0
    assert judge.calls == 0
