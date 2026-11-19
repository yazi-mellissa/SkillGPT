"""Measure the three SkillGPT backends on the same queries.

    python -m benchmark.run_benchmark --url http://localhost:8000
    python -m benchmark.run_benchmark --backends vector openai
    python -m benchmark.run_benchmark --mock        # no service needed, self-test

Writes a table to stdout and the full per-query detail to benchmark/results.json.

Caching: the service caches responses, so a second run of the same queries measures
the cache, not the backend. Start the service with USE_GPTCACHE=false, or pass
--cache-warmup to run every query once before timing so that all three backends are
measured in the same, warm state. Reporting a cold LLM against a warm index would be
the easiest way to get a wrong answer out of this script.
"""

import argparse
import json
import time
from pathlib import Path

from .metrics import BackendReport, QueryScore, format_table, hardest_queries
from .normalize import extract_llm_skills, extract_vector_skills, normalize_set

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
RESULTS = HERE / "results.json"

BACKENDS = {
    "vector": {"path": "/search", "extract": extract_vector_skills},
    "openai": {"path": "/search_openai", "extract": extract_llm_skills},
    "gemini": {"path": "/search_gemini", "extract": extract_llm_skills},
}

# Published list prices at the time of writing, per million tokens, for the models
# this service defaults to. They are an input to the script, not a measurement: check
# them against the provider's current pricing page before quoting the dollar column.
PRICING_USD_PER_MTOK = {
    "openai": {"input": 0.15, "output": 0.60},
    "gemini": {"input": 0.075, "output": 0.30},
}
ASSUMED_TOKENS = {"input": 350, "output": 60}


def load_dataset(path=DATASET) -> list:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def estimate_cost_per_1k(backend: str) -> float:
    pricing = PRICING_USD_PER_MTOK.get(backend)
    if not pricing:
        return 0.0  # the vector index costs nothing per query
    per_query = (
        ASSUMED_TOKENS["input"] * pricing["input"]
        + ASSUMED_TOKENS["output"] * pricing["output"]
    ) / 1_000_000
    return round(per_query * 1000, 4)


def query_backend(session, base_url: str, path: str, text: str, timeout: float):
    """Call one endpoint and return (payload, latency_ms, error)."""
    started = time.perf_counter()
    try:
        response = session.get(f"{base_url}{path}", params={"query": text},
                               timeout=timeout)
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return None, elapsed, f"HTTP {response.status_code}"
        return response.json(), elapsed, ""
    except Exception as exc:  # noqa: BLE001 - any failure is a failed query
        return None, (time.perf_counter() - started) * 1000, type(exc).__name__


def mock_backend(backend: str, row: dict):
    """A stand-in used by --mock, so the harness can be exercised without a service.

    Each backend is given a deliberately different failure mode, which is what makes
    the mock run a real test of the scoring rather than a formality:
      vector  perfect recall is impossible, it drops the last gold skill and adds a
              plausible neighbour from its fixed vocabulary
      openai  recovers everything but also invents one skill
      gemini  recovers everything and writes two of them in another spelling
    """
    gold = list(row["skills"])
    if backend == "vector":
        return gold[:-1] + ["agile"] if gold else ["agile"]
    if backend == "openai":
        return gold + ["communication"]
    spellings = {"javascript": "JS", "kubernetes": "K8s", "node": "NodeJS",
                 "python": "Py", "postgresql": "Postgres"}
    return [spellings.get(s, s) for s in gold]


def evaluate(backend: str, rows: list, base_url: str, timeout: float,
             mock: bool, warmup: bool) -> BackendReport:
    config = BACKENDS[backend]
    report = BackendReport(name=backend, cost_per_1k_usd=estimate_cost_per_1k(backend))

    session = None
    if not mock:
        import requests

        session = requests.Session()
        if warmup:
            for row in rows:
                query_backend(session, base_url, config["path"], row["text"], timeout)

    for row in rows:
        expected = normalize_set(row["skills"])

        if mock:
            started = time.perf_counter()
            payload = mock_backend(backend, row)
            latency = (time.perf_counter() - started) * 1000
            error = ""
        else:
            payload, latency, error = query_backend(
                session, base_url, config["path"], row["text"], timeout
            )

        predicted = set() if error else normalize_set(config["extract"](payload))
        report.scores.append(
            QueryScore(query_id=row["id"], predicted=predicted, expected=expected,
                       latency_ms=latency, error=error)
        )
        print(f"  {backend:7s} {row['id']:8s} "
              f"{'ERROR ' + error if error else f'F1 {report.scores[-1].f1:.2f}'}")

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000",
                        help="base URL of a running SkillGPT service")
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS),
                        choices=list(BACKENDS))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--mock", action="store_true",
                        help="run against synthetic responses, no service required")
    parser.add_argument("--cache-warmup", action="store_true",
                        help="query everything once before timing, so every backend "
                             "is measured warm")
    args = parser.parse_args(argv)

    rows = load_dataset()
    print(f"{len(rows)} annotated queries, "
          f"{sum(len(r['skills']) for r in rows)} gold skills\n")

    reports = []
    for backend in args.backends:
        print(f"{backend}:")
        reports.append(evaluate(backend, rows, args.url, args.timeout,
                                args.mock, args.cache_warmup))
        print()

    print(format_table(reports))
    print()

    for report in reports:
        if not report.answered:
            continue
        print(f"{report.name}, worst queries:")
        for score in hardest_queries(report, 3):
            print(f"  {score.query_id} F1 {score.f1:.2f}")
            if score.false_negatives:
                print(f"    missed   : {sorted(score.false_negatives)}")
            if score.false_positives:
                print(f"    invented : {sorted(score.false_positives)}")
        print()

    RESULTS.write_text(json.dumps({
        "dataset": str(DATASET.name),
        "queries": len(rows),
        "mock": args.mock,
        "summary": [r.row() for r in reports],
        "detail": {
            r.name: [
                {
                    "id": s.query_id,
                    "expected": sorted(s.expected),
                    "predicted": sorted(s.predicted),
                    "f1": round(s.f1, 4),
                    "latency_ms": round(s.latency_ms, 1),
                    "error": s.error,
                }
                for s in r.scores
            ]
            for r in reports
        },
    }, indent=2), encoding="utf-8")
    print(f"detail written to {RESULTS.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
