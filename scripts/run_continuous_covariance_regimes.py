#!/usr/bin/env python3
"""Run the continuous covariance experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bwar.paper_jcgs.continuous_covariance_regimes import (  # noqa: E402
    ContinuousCovarianceConfig,
    decision_table,
    run_replication,
    summarize_performance,
)


def _worker(job):
    return run_replication(*job)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = ContinuousCovarianceConfig.from_mapping(payload["design"])
    reps = int(payload.get("replications", 20) if args.reps is None else args.reps)
    if reps < 1:
        raise ValueError("reps must be positive")
    result_dir = args.output.resolve()
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    jobs = [(replication, config) for replication in range(reps)]
    iterator = map(_worker, jobs)
    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(_worker, jobs)
    performance, origins, references = [], [], []
    try:
        for completed, result in enumerate(iterator, start=1):
            performance.append(result[0])
            origins.append(result[1])
            references.append(result[2])
            if completed == 1 or completed % max(1, reps // 10) == 0 or completed == reps:
                print(f"[continuous-covariance] completed {completed}/{reps}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()

    raw = pd.concat(performance, ignore_index=True)
    raw = raw.loc[
        raw["method"].isin(
            (
                "persistence",
                "euclidean",
                "cholesky",
                "log_euclidean",
                "fixed",
                "local",
            )
        )
    ].copy()
    origin = pd.concat(origins, ignore_index=True)
    reference = pd.concat(references, ignore_index=True)
    summary = summarize_performance(raw)
    decision = decision_table(raw)
    raw.to_csv(result_dir / "replication_results.csv.gz", index=False, compression="gzip")
    origin.to_csv(result_dir / "origin_results.csv.gz", index=False, compression="gzip")
    reference.to_csv(result_dir / "reference_diagnostics.csv.gz", index=False, compression="gzip")
    summary.to_csv(result_dir / "summary.csv", index=False)
    decision.to_csv(result_dir / "decision.csv", index=False)
    try:
        config_entry = str(args.config.resolve().relative_to(ROOT))
    except ValueError:
        config_entry = args.config.name
    protocol = {
        "experiment": payload.get("experiment"),
        "config": config_entry,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "replications": reps,
        "methods": [
            "persistence",
            "euclidean",
            "cholesky",
            "log_euclidean",
            "fixed",
            "local",
        ],
        "ridge_selection": "method-specific validation loss",
        "paired_replications": True,
    }
    (result_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"elapsed_seconds={time.perf_counter() - started:.2f}", flush=True)
    print(decision.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
