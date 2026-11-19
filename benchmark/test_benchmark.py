"""Tests for the parts of the benchmark that can be checked without a service.

The scoring and the normalisation decide every number this harness produces, so they
are the parts that must not be wrong. Run with `pytest benchmark/`.
"""

from .metrics import BackendReport, QueryScore
from .normalize import (
    extract_llm_skills,
    extract_vector_skills,
    normalize_set,
    normalize_skill,
)


def test_normalisation_folds_case_accents_and_punctuation():
    assert normalize_skill("  Python.  ") == "python"
    assert normalize_skill("Modelisation") == normalize_skill("Modélisation")
    assert normalize_skill("C++") == "c++"
    assert normalize_skill("CI/CD") == "ci/cd"


def test_aliases_collapse_spellings_of_one_skill():
    assert normalize_skill("NodeJS") == normalize_skill("Node.js") == "node"
    assert normalize_skill("K8s") == "kubernetes"
    assert normalize_skill("Postgres") == "postgresql"


def test_empty_and_whitespace_skills_are_dropped():
    assert normalize_set(["Python", "", "  ", None]) == {"python"}


def test_vector_payload_is_read_in_every_shape_pymilvus_uses():
    nested = [[{"id": 1, "distance": 0.2, "entity": {"expertise": "Python"}}]]
    flat = [{"entity": {"text": "Python"}}]
    assert extract_vector_skills(nested) == ["Python"]
    assert extract_vector_skills(flat) == ["Python"]
    assert extract_vector_skills(["Python"]) == ["Python"]
    assert extract_vector_skills(None) == []


def test_llm_payload_split_when_the_model_returns_one_string():
    assert extract_llm_skills(["Python, SQL"]) == ["Python", " SQL"]
    assert extract_llm_skills("Python, SQL") == ["Python", " SQL"]


def test_precision_and_recall_separate_the_two_failure_modes():
    # Recovers everything, invents one: recall 1, precision 2/3.
    invents = QueryScore("a", {"python", "sql", "agile"}, {"python", "sql"}, 10.0)
    assert invents.recall == 1.0
    assert round(invents.precision, 4) == round(2 / 3, 4)

    # Misses one, invents nothing: precision 1, recall 1/2.
    misses = QueryScore("b", {"python"}, {"python", "sql"}, 10.0)
    assert misses.precision == 1.0
    assert misses.recall == 0.5

    # F1 alone ranks the backend that invents a skill above the one that misses one.
    # Whether that is the right ranking depends on what the caller does with the
    # answer, so the table reports precision and recall next to it rather than
    # settling the question silently.
    assert invents.f1 > misses.f1


def test_an_empty_answer_against_a_non_empty_gold_scores_zero():
    score = QueryScore("c", set(), {"python"}, 10.0)
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_the_noise_query_rewards_returning_nothing():
    # Gold is empty. Returning nothing is correct; recall is defined as 1 so the
    # query does not silently punish the only right answer.
    score = QueryScore("edge-04", set(), set(), 10.0)
    assert score.recall == 1.0


def test_errored_queries_are_excluded_from_the_averages_but_counted():
    report = BackendReport("x")
    report.scores = [
        QueryScore("a", {"python"}, {"python"}, 10.0),
        QueryScore("b", set(), {"sql"}, 5.0, error="Timeout"),
    ]
    assert report.errors == 1
    assert report.macro_f1 == 1.0  # the timeout does not count as a score of zero
    assert len(report.answered) == 1


def test_micro_and_macro_f1_differ_when_query_length_varies():
    report = BackendReport("x")
    report.scores = [
        QueryScore("short", {"python"}, {"python"}, 1.0),
        QueryScore("long", {"a", "b"}, {"a", "b", "c", "d", "e", "f"}, 1.0),
    ]
    assert report.macro_f1 != report.micro_f1


def test_latency_percentiles():
    report = BackendReport("x")
    report.scores = [
        QueryScore(str(i), {"python"}, {"python"}, float(i)) for i in range(1, 101)
    ]
    assert report.latency_p50_ms == 50.5
    assert report.latency_p95_ms == 96.0
