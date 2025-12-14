"""
Integration tests for the SyntraLine++ CLI (`syntra` binary).

These tests exercise the full pipeline:
    .syntra file  ->  C++ compiler  ->  diagnostics / IR / simulation

They assume:
    - The project has been built (build/src/syntra exists)
    - Pytest is run from the project root (or ROOT detection below works)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def get_project_root() -> Path:
    # tests/integration/test_cli_integration.py
    #   -> tests/integration
    #   -> tests
    #   -> project root
    return Path(__file__).resolve().parents[2]


def get_syntra_binary() -> Path:
    root = get_project_root()
    syntra = root / "build" / "src" / "syntra"
    return syntra


def run_syntra(args: List[str]) -> Tuple[int, str, str]:
    """
    Run the `syntra` binary with the given arguments.

    Returns:
        (exit_code, stdout, stderr)
    """
    syntra = get_syntra_binary()
    if not syntra.exists():
        pytest.skip(f"syntra binary not found at {syntra}; run ./scripts/build.sh first")

    cmd = [str(syntra)] + args
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_mnist_basic_dump_ast_and_ir():
    """
    Run `syntra --dump-ast` and `syntra --dump-ir` on examples/mnist_basic.syntra
    and check that the output looks structurally correct.
    """
    root = get_project_root()
    mnist = root / "examples" / "mnist_basic.syntra"
    assert mnist.exists(), f"Example file not found: {mnist}"

    # AST dump
    code, out, err = run_syntra(["--dump-ast", str(mnist)])
    assert code == 0, f"--dump-ast failed: {err}"
    # Expect the dataset, model and experiment names to appear
    assert "dataset mnist" in out
    assert "model simple_cnn" in out
    assert "experiment cnn_mnist" in out

    # IR dump
    code, out, err = run_syntra(["--dump-ir", str(mnist)])
    assert code == 0, f"--dump-ir failed: {err}"
    # Basic structural markers from dumpIR()
    assert "Datasets (" in out
    assert "Models (" in out
    assert "Experiments (" in out
    assert "cnn_mnist" in out


def test_mnist_basic_simulate():
    """
    Run `syntra --simulate` on mnist_basic and check that simulation output is printed.
    """
    root = get_project_root()
    mnist = root / "examples" / "mnist_basic.syntra"
    assert mnist.exists(), f"Example file not found: {mnist}"

    code, out, err = run_syntra(["--simulate", str(mnist)])

    assert code == 0, f"--simulate failed on mnist_basic: {err}"
    # The simulation driver prints something like:
    #   Simulated experiment 'cnn_mnist':
    assert "Simulated experiment 'cnn_mnist'" in out
    # And some metrics lines
    assert "loss =" in out
    assert "accuracy =" in out


def test_no_experiment_error():
    """
    Run `syntra --simulate` on a program that has no experiments.
    Expect a non-zero exit code and a clear diagnostic.
    """
    root = get_project_root()
    src = root / "tests" / "testdata" / "no_experiment.syntra"
    assert src.exists(), f"Testdata file not found: {src}"

    code, out, err = run_syntra(["--simulate", str(src)])

    # We expect an error (no experiments defined in module)
    assert code != 0, "Expected non-zero exit code for no-experiment program"
    # The error message was observed earlier in run_examples.sh output
    assert "no experiments defined in module" in err


def test_syntax_error_reports_diagnostic():
    """
    Run `syntra --dump-ast` on a syntactically invalid program.
    Expect a non-zero exit code and a diagnostic mentioning an error.
    """
    root = get_project_root()
    src = root / "tests" / "testdata" / "syntax_error.syntra"
    assert src.exists(), f"Testdata file not found: {src}"

    code, out, err = run_syntra(["--dump-ast", str(src)])

    # Parser should fail and diagnostics should be emitted
    assert code != 0, "Expected non-zero exit code for syntax error program"
    assert "error:" in err or "expected" in err.lower()
