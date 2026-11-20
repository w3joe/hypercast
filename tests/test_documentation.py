"""Regression checks for docstring coverage, doctests, OpenAPI, and MkDocs."""

from __future__ import annotations

import ast
import doctest
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

import hypercast4d
from hypercast4d.playground import create_app


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "hypercast4d"


def test_public_package_exports_are_documented() -> None:
    assert hypercast4d.__doc__ and "HyperCAST" in hypercast4d.__doc__
    assert len(hypercast4d.__all__) >= 20
    for symbol in hypercast4d.__all__:
        obj = getattr(hypercast4d, symbol)
        assert obj is not None
        if callable(obj) or isinstance(obj, type):
            assert obj.__doc__, f"Exported symbol {symbol!r} is missing a docstring"


def test_all_public_modules_classes_and_functions_have_docstrings() -> None:
    missing: list[str] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(f"{path.name}: module docstring")
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    missing.append(f"{path.name}:{node.name}")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                        if not ast.get_docstring(item):
                            missing.append(f"{path.name}:{node.name}.{item.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    missing.append(f"{path.name}:{node.name}")
    assert not missing, f"Undocumented public symbols: {missing}"


@pytest.mark.parametrize(
    "module_name",
    [
        "hypercast4d",
        "hypercast4d.algebras",
        "hypercast4d.layers",
        "hypercast4d.multilayer_hyperdense",
        "hypercast4d.architecture",
        "hypercast4d.graph_architecture",
        "hypercast4d.data",
        "hypercast4d.training",
    ],
)
def test_docstring_examples_execute_cleanly(module_name: str) -> None:
    module = importlib.import_module(module_name)
    failures, attempted = doctest.testmod(module, verbose=False, optionflags=doctest.ELLIPSIS)
    assert failures == 0
    assert attempted >= 1


def test_openapi_schema_documents_api_v1_endpoints(tmp_path: Path) -> None:
    app = create_app(results_root=tmp_path / "playground", project_root=REPO_ROOT)
    schema = app.openapi()
    assert schema["info"]["title"] == "Hypercast Playground API"
    api_paths = {p: ops for p, ops in schema["paths"].items() if p.startswith("/api/v1/")}
    assert len(api_paths) >= 12
    for path, methods in api_paths.items():
        for method, operation in methods.items():
            assert operation.get("summary"), f"{method.upper()} {path} missing summary"
            assert operation.get("tags"), f"{method.upper()} {path} missing tags"
            assert operation.get("description"), f"{method.upper()} {path} missing description"


def test_mkdocs_strict_build(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs")
    pytest.importorskip("mkdocstrings")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--config-file",
            str(REPO_ROOT / "mkdocs.yml"),
            "--site-dir",
            str(tmp_path / "site"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"mkdocs build --strict failed:\n{result.stderr}"
    assert (tmp_path / "site" / "index.html").is_file()
    assert (tmp_path / "site" / "api" / "algebras_and_layers" / "index.html").is_file()
