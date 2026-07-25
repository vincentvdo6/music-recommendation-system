"""Release-artifact installation safety checks."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from services.recommendation.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _candidate_zip(path: Path, gate: dict) -> Path:
    required = {
        "lightgbm_ranker_v2.txt": b"model",
        "ncf_item_v2.pt": b"ncf",
        "track_meta.parquet": b"meta",
        "metrics.json": json.dumps({"decision_gate": gate}).encode(),
        "feature_names.json": json.dumps(FEATURE_NAMES).encode(),
        "eval_sample.parquet": b"eval",
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in required.items():
            zf.writestr(name, payload)
    return path


def test_installer_refuses_failed_release_before_writing(tmp_path):
    archive = _candidate_zip(
        tmp_path / "failed.zip",
        {"pass": False, "ndcg_diff": -0.01, "ndcg_ci": [-0.02, -0.001]},
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "install_artifacts.py"), str(archive)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing artifact" in result.stdout
    assert not (tmp_path / "models").exists()


def test_installer_requires_an_explicit_completed_gate(tmp_path):
    archive = _candidate_zip(tmp_path / "unknown.zip", {})
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "install_artifacts.py"), str(archive)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "failed/missing" not in result.stdout
    assert "did not pass" in result.stdout
