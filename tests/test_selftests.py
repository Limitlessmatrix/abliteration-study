"""Run the library's own self-tests (scripts/selftest*.py and the --selftest modes) under pytest.

Each self-test is a standalone script that builds tiny random models and prints PASS/FAIL lines; it exits non-zero
on any failure. Notes:
  * scripts/selftest.py and scripts/selftest_big.py load the Qwen2.5-1.5B-Instruct TOKENIZER from the Hugging Face
    Hub (a few MB, no weights). They are skipped when the Hub is unreachable and the tokenizer is not cached.
  * the others need only torch (and optuna for the search tests).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(args, timeout=1200):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _have_tokenizer() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        return True
    except Exception:
        return False


@pytest.mark.parametrize("script", ["selftest_grid.py", "selftest_guard.py", "selftest_v3search.py"])
def test_offline_selftests(script):
    pytest.importorskip("torch")
    if script != "selftest_guard.py":
        pytest.importorskip("optuna")
    code, out = _run([f"scripts/{script}"])
    assert code == 0, out[-3000:]


@pytest.mark.parametrize("script", ["selftest.py", "selftest_big.py"])
def test_tokenizer_selftests(script):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not _have_tokenizer():
        pytest.skip("Qwen2.5-1.5B-Instruct tokenizer not cached and the Hub is unreachable")
    code, out = _run([f"scripts/{script}"])
    assert code == 0, out[-3000:]


def test_export_adapter_selftest():
    pytest.importorskip("torch")
    code, out = _run(["scripts/export_adapter.py", "--selftest"])
    assert code == 0 and "13/13 pass" in out, out[-3000:]


def test_repalign_selftest():
    pytest.importorskip("torch")
    code, out = _run(["eval/repalign.py", "--selftest"])
    assert code == 0, out[-3000:]
