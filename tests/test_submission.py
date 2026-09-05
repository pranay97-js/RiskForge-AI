"""Submission verification suite: checks all Section 12.1 submission requirements."""

from __future__ import annotations

from pathlib import Path
import pytest


def test_model_artifacts_exist():
    required_models = [
        "models/baseline.pkl",
        "models/xgboost.pkl",
        "models/feature_pipeline.pkl",
        "models/metadata.json",
    ]
    for m in required_models:
        path = Path(m)
        assert path.exists(), f"Required model artifact missing: {m}"
        assert path.stat().st_size > 0, f"Model artifact is empty: {m}"


def test_frozen_test_partition_exists():
    path = Path("data/processed/test_frozen.csv")
    assert path.exists(), "Frozen test set data/processed/test_frozen.csv missing!"
    assert path.stat().st_size > 1000, "Frozen test set is too small or empty!"


def test_env_example_no_secrets():
    env_ex = Path(".env.example")
    assert env_ex.exists(), ".env.example missing!"
    content = env_ex.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in content
    # Ensure no real secret key has been hardcoded
    assert "sk-" not in content, "Found potential real API key in .env.example!"


def test_defense_only_safety_boundary():
    # Verify no offensive fraud evasion or card theft capabilities in codebase
    src_dir = Path("src")
    forbidden_terms = ["generate_stolen_cards", "bypass_gateway", "evasion_attack"]

    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore").lower()
        for term in forbidden_terms:
            assert term not in text, f"Forbidden offensive term '{term}' found in {py_file}!"


def test_documentation_deliverables_exist():
    required_docs = [
        "README.md",
        "reports/model_card.md",
        "requirements.txt",
        "app.py",
    ]
    for doc in required_docs:
        p = Path(doc)
        assert p.exists(), f"Required deliverable missing: {doc}"
        assert p.stat().st_size > 0, f"Deliverable is empty: {doc}"
