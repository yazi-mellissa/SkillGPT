"""Compare BERT, RoBERTa and DistilBERT on the skill-extraction set.

    python -m benchmark.compare_encoders
    python -m benchmark.compare_encoders --models bert distilbert --top-k 5

No service and no API key: the encoders are called in process, so this runs on a laptop
and costs nothing. That is the point of including a local backend at all, and it is what
makes this comparison repeatable by whoever reads the repository.

What is being measured. Each model embeds the skill catalogue once, then embeds each
query and returns its nearest entries. The models share the catalogue, the pooling and
the scoring, so any difference between them comes from pre-training alone.

The threshold sweep matters more than the headline number. A nearest-neighbour backend
always returns `top_k` answers, so precision is bounded by how many of them are right; it
is the minimum similarity that decides whether the backend is cautious or generous, and
the same model looks very different at either end of that sweep.
"""

import argparse
import json
import time
from pathlib import Path

from .metrics import BackendReport, QueryScore, format_table
from .normalize import normalize_set
from .run_benchmark import DATASET, load_dataset

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "encoder_results.json"


_SERVICES: dict = {}


def _service(model_key: str):
    """One instance per model: loading the weights and embedding the catalogue is the
    expensive part, and it must not be charged to every threshold in the sweep."""
    from app.services.encoder_service import EncoderService

    if model_key not in _SERVICES:
        started = time.perf_counter()
        _SERVICES[model_key] = (EncoderService(model_key=model_key),
                                round(time.perf_counter() - started, 1))
    return _SERVICES[model_key]


def evaluate(model_key: str, rows: list, top_k: int, min_score: float,
             mode: str) -> tuple:
    """Return (report, load_seconds) for one encoder at one threshold, in one mode."""
    service, load_seconds = _service(model_key)

    label = f"{model_key}/{mode} t={min_score:g}"
    report = BackendReport(name=label, cost_per_1k_usd=0.0)
    for row in rows:
        expected = normalize_set(row["skills"])
        t0 = time.perf_counter()
        try:
            matches = service.search(row["text"], top_k=top_k,
                                     min_score=min_score, mode=mode)
            error = ""
        except Exception as exc:  # noqa: BLE001
            matches, error = [], type(exc).__name__
        latency = (time.perf_counter() - t0) * 1000

        predicted = normalize_set(m["skill"] for m in matches) if not error else set()
        report.scores.append(
            QueryScore(query_id=row["id"], predicted=predicted, expected=expected,
                       latency_ms=latency, error=error)
        )
    return report, load_seconds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["bert", "roberta", "distilbert"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.85, 0.92],
                    help="minimum cosine similarity; one run per value")
    ap.add_argument("--modes", nargs="+", default=["spans", "document"],
                    choices=["spans", "document"])
    args = ap.parse_args(argv)

    rows = load_dataset(DATASET)
    print(f"{len(rows)} annotated queries, "
          f"{sum(len(r['skills']) for r in rows)} gold skills, top_k={args.top_k}\n")

    reports, loads = [], {}
    for model_key in args.models:
        for mode in args.modes:
            for threshold in args.thresholds:
                print(f"  {model_key} / {mode} / t={threshold:g} ...", flush=True)
                report, load_seconds = evaluate(model_key, rows, args.top_k,
                                                threshold, mode)
                reports.append(report)
                loads.setdefault(model_key, load_seconds)
    print()

    print(format_table(reports))
    print()
    print("Model load and catalogue embedding, once per process:")
    for model_key, seconds in loads.items():
        print(f"  {model_key:12s} {seconds:5.1f} s")

    RESULTS.write_text(json.dumps({
        "queries": len(rows),
        "top_k": args.top_k,
        "thresholds": args.thresholds,
        "modes": args.modes,
        "load_seconds": loads,
        "summary": [r.row() for r in reports],
    }, indent=2), encoding="utf-8")
    print(f"\ndetail written to {RESULTS.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
