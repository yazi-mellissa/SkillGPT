"""Scoring one backend against the annotated set.

Three things are measured, because no single one decides which backend to use:

  quality  precision, recall and F1 against the gold skills
  latency  p50 and p95 wall clock per query, which is what a caller feels
  cost     US dollars per thousand queries, zero for the vector index

Precision and recall split the two failure modes apart. The vector backend cannot
invent a skill, so its precision is bounded by its vocabulary; an LLM can name a skill
the index has never heard of, and can also name one that is not in the text at all.
A single F1 would hide exactly the difference the benchmark exists to show.
"""

from dataclasses import dataclass, field
from statistics import median


@dataclass
class QueryScore:
    query_id: str
    predicted: set
    expected: set
    latency_ms: float
    error: str = ""

    @property
    def true_positives(self) -> set:
        return self.predicted & self.expected

    @property
    def false_positives(self) -> set:
        return self.predicted - self.expected

    @property
    def false_negatives(self) -> set:
        return self.expected - self.predicted

    @property
    def precision(self) -> float:
        """Share of the returned skills that are in the text.

        With nothing returned and nothing expected the answer is right, so precision
        is 1 rather than 0. The noise query in the dataset, an office description
        with no skills in it, is exactly this case, and scoring it zero would punish
        the only correct behaviour.
        """
        if not self.predicted:
            return 1.0 if not self.expected else 0.0
        return len(self.true_positives) / len(self.predicted)

    @property
    def recall(self) -> float:
        """Share of the skills in the text that were returned.

        Vacuously 1 when there is nothing to find.
        """
        if not self.expected:
            return 1.0
        return len(self.true_positives) / len(self.expected)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class BackendReport:
    name: str
    scores: list = field(default_factory=list)
    cost_per_1k_usd: float = 0.0

    @property
    def answered(self) -> list:
        return [s for s in self.scores if not s.error]

    @property
    def errors(self) -> int:
        return sum(1 for s in self.scores if s.error)

    def _mean(self, attribute: str) -> float:
        rows = self.answered
        return sum(getattr(s, attribute) for s in rows) / len(rows) if rows else 0.0

    @property
    def macro_precision(self) -> float:
        return self._mean("precision")

    @property
    def macro_recall(self) -> float:
        return self._mean("recall")

    @property
    def macro_f1(self) -> float:
        return self._mean("f1")

    @property
    def micro_f1(self) -> float:
        """Pooled over all queries, so a long document counts more than a short one.

        Reported next to the macro score because the two diverge when a backend is
        good on short job titles and poor on full CVs, which is worth knowing.
        """
        tp = sum(len(s.true_positives) for s in self.answered)
        fp = sum(len(s.false_positives) for s in self.answered)
        fn = sum(len(s.false_negatives) for s in self.answered)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @property
    def latency_p50_ms(self) -> float:
        rows = [s.latency_ms for s in self.answered]
        return median(rows) if rows else 0.0

    @property
    def latency_p95_ms(self) -> float:
        rows = sorted(s.latency_ms for s in self.answered)
        if not rows:
            return 0.0
        return rows[min(int(0.95 * len(rows)), len(rows) - 1)]

    def row(self) -> dict:
        return {
            "backend": self.name,
            "macro F1": round(self.macro_f1, 4),
            "micro F1": round(self.micro_f1, 4),
            "precision": round(self.macro_precision, 4),
            "recall": round(self.macro_recall, 4),
            "p50 ms": round(self.latency_p50_ms, 1),
            "p95 ms": round(self.latency_p95_ms, 1),
            "$/1k": self.cost_per_1k_usd,
            "errors": self.errors,
        }


def hardest_queries(report: BackendReport, n: int = 5) -> list:
    """The queries a backend did worst on, which is where the failure mode shows."""
    return sorted(report.answered, key=lambda s: s.f1)[:n]


def format_table(reports: list) -> str:
    rows = [r.row() for r in reports]
    if not rows:
        return "(no results)"
    columns = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    separator = "  ".join("-" * widths[c] for c in columns)
    body = [
        "  ".join(str(r[c]).ljust(widths[c]) for c in columns)
        for r in sorted(rows, key=lambda r: -r["macro F1"])
    ]
    return "\n".join([header, separator, *body])
