from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
S2_METHODS = {
    "persistence",
    "euclidean",
    "cholesky",
    "log_euclidean",
    "fixed",
    "local",
}
PEMS_METHODS = {
    "Persistence",
    "Euclidean",
    "Cholesky",
    "Log-Euclidean",
    "BWAR",
}


class ArticleArtifactTests(unittest.TestCase):
    def test_tables_rebuild_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_artifacts.py"),
                    "--output-root",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
            for name in (
                "geometry_robustness_rebuild.tex",
                "divvy_full_results_rebuild.tex",
                "pems_full_results.tex",
            ):
                self.assertEqual(
                    (output / "tables" / name).read_bytes(),
                    (ROOT / "artifacts" / "reference" / "tables" / name).read_bytes(),
                )

    def test_reference_panels_match_article_protocol(self) -> None:
        s1 = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "s1_geometry"
            / "strong_synthetic_transport_summary.csv"
        )
        self.assertEqual(set(s1["n_rep"]), {80})

        s2 = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "s2_continuous_covariance"
            / "summary.csv"
        )
        self.assertEqual(set(s2["n_replications"]), {60})
        self.assertEqual(set(s2["regime"]), {"persistent", "mean_reverting"})
        self.assertEqual(set(s2["method"]), S2_METHODS)
        s2_protocol = json.loads(
            (
                ROOT
                / "results"
                / "reference"
                / "s2_continuous_covariance"
                / "protocol.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(set(s2_protocol["methods"]), S2_METHODS)
        config_path = ROOT / s2_protocol["config"]
        self.assertEqual(
            hashlib.sha256(config_path.read_bytes()).hexdigest(),
            s2_protocol["config_sha256"],
        )
        self.assertEqual(
            s2_protocol["ridge_selection"], "method-specific validation loss"
        )
        s2_raw = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "s2_continuous_covariance"
            / "replication_results.csv.gz"
        )
        self.assertEqual(set(s2_raw["method"]), S2_METHODS)
        self.assertEqual(set(s2_raw["replication"]), set(range(60)))
        checks = json.loads(
            (
                ROOT
                / "results"
                / "reference"
                / "s2_continuous_covariance"
                / "integrity_checks.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(checks["passed"])

        divvy = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "divvy"
            / "target_level_losses.csv"
        )
        self.assertEqual(set(divvy["horizon"]), {3, 4, 5})
        self.assertEqual(divvy["method"].nunique(), 7)
        protocol = json.loads(
            (
                ROOT
                / "results"
                / "reference"
                / "divvy"
                / "protocol.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(set(protocol["methods"]), set(divvy["method"]))
        self.assertEqual(
            protocol["selection_endpoint"],
            "training-standardized station-mean RMSE",
        )
        targets = divvy.groupby(["method", "horizon"])["target_index"].nunique()
        self.assertEqual(set(targets), {183})

        pems = pd.read_csv(
            ROOT
            / "results"
            / "reference"
            / "pems_bay"
            / "test_method_summary.csv"
        )
        self.assertEqual(set(pems["horizon"]), {1, 3, 6})
        self.assertEqual(set(pems["method"]), PEMS_METHODS)

    def test_release_tree_excludes_superseded_workflows(self) -> None:
        release_roots = (
            ROOT / "configs",
            ROOT / "scripts",
            ROOT / "src",
            ROOT / "results" / "reference",
            ROOT / "artifacts" / "reference",
        )
        retired_tokens = {
            "beijing",
            "electricity",
            "weekly",
            "seasonal",
            "estimated_states",
            "theory_diagnostics",
            "rolling_drift",
            "local_shared",
            "matched_reference",
        }
        for root in release_roots:
            for path in root.rglob("*"):
                normalized = str(path.relative_to(ROOT)).lower().replace("-", "_")
                self.assertFalse(
                    any(token in normalized for token in retired_tokens),
                    f"superseded release path retained: {normalized}",
                )


if __name__ == "__main__":
    unittest.main()
