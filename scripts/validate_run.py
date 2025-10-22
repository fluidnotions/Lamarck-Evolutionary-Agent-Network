#!/usr/bin/env python
"""Quick QA script to validate reasoning retention and score progression."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean

from lean.evaluation import ContentEvaluator


THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
FINAL_PATTERN = re.compile(r"<final>.*?</final>", re.IGNORECASE | re.DOTALL)
AVERAGE_PATTERN = re.compile(r"Average:\s*([0-9]+(?:\.[0-9]+)?)\/10")


def summarize_log(log_text: str) -> dict[str, float | int]:
    """Compute reasoning coverage and score statistics from a log file."""
    think_blocks = THINK_PATTERN.findall(log_text)
    final_blocks = FINAL_PATTERN.findall(log_text)
    averages = [float(match) for match in AVERAGE_PATTERN.findall(log_text)]

    coverage = (len(think_blocks) / len(final_blocks) * 100.0) if final_blocks else 0.0
    avg_score = mean(averages) if averages else 0.0
    score_delta = averages[-1] - averages[0] if len(averages) >= 2 else 0.0

    return {
        "reasoning_coverage": coverage,
        "total_responses": len(final_blocks),
        "avg_score": avg_score,
        "score_delta": score_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LEAN experiment output")
    parser.add_argument(
        "--log-dir",
        default="./logs",
        help="Directory containing LEAN log files (default: ./logs)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    log_files = sorted(log_dir.glob("lean_*.log"))
    if not log_files:
        raise SystemExit("No LEAN log files found. Run an experiment first.")

    latest_log = max(log_files, key=lambda path: path.stat().st_mtime)
    log_text = latest_log.read_text(encoding="utf-8", errors="ignore")

    metrics = summarize_log(log_text)
    evaluator = ContentEvaluator()  # Ensures evaluation stack imports correctly

    print("QA SUMMARY")
    print("----------")
    print(f"Log file: {latest_log.name}")
    print(f"Reasoning coverage: {metrics['reasoning_coverage']:.1f}%")
    print(f"Responses analyzed: {metrics['total_responses']}")
    print(f"Average score: {metrics['avg_score']:.2f}/10")
    print(f"Score delta (first → last): {metrics['score_delta']:.2f}")
    print("Evaluator ready:", isinstance(evaluator, ContentEvaluator))


if __name__ == "__main__":
    main()
