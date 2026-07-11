from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

UI_ASSETS = ("index.html", "app.css", "app.js")
BENCHMARK_ASSETS = (
    "commit-review-benchmark.md",
    "commit_review/suite.json",
    "commit_review/android_exported_webview/after/app/src/main/AndroidManifest.xml",
    "commit_review/ios_background_ui/after/App/ProfileViewController.swift",
    "commit_review/cpp_dangling_view/after/src/config.cpp",
    "commit_review/django_injection/after/projects/views.py",
    "commit_review/pytorch_invalid_eval/after/experiment.py",
)


def main() -> int:
    distribution_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = sorted(distribution_dir.glob("*.whl"))
    source_archives = sorted(distribution_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(source_archives) != 1:
        raise RuntimeError("expected exactly one wheel and one source archive")

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = set(archive.namelist())
    missing_wheel = [name for name in UI_ASSETS if f"inverse_agent/ui/{name}" not in wheel_names]
    missing_wheel_benchmark = [
        name
        for name in BENCHMARK_ASSETS
        if f"inverse_agent/benchmark_assets/{name}" not in wheel_names
    ]
    unexpected_wheel_cache = [
        name for name in wheel_names if "/__pycache__/" in name or name.endswith((".pyc", ".pyo"))
    ]

    with tarfile.open(source_archives[0], "r:gz") as archive:
        source_names = {member.name for member in archive.getmembers()}
    missing_source = [
        name
        for name in UI_ASSETS
        if not any(path.endswith(f"/src/inverse_agent/ui/{name}") for path in source_names)
    ]
    missing_source_benchmark = [
        name
        for name in BENCHMARK_ASSETS
        if not any(
            path.endswith(f"/src/inverse_agent/benchmark_assets/{name}") for path in source_names
        )
    ]
    missing_source_checkout = [
        path
        for path in ("benchmarks/commit_review/suite.json", "docs/commit-review-benchmark.md")
        if not any(name.endswith(f"/{path}") for name in source_names)
    ]
    unexpected_source_cache = [
        name for name in source_names if "/__pycache__/" in name or name.endswith((".pyc", ".pyo"))
    ]

    if (
        missing_wheel
        or missing_source
        or missing_wheel_benchmark
        or missing_source_benchmark
        or missing_source_checkout
        or unexpected_wheel_cache
        or unexpected_source_cache
    ):
        raise RuntimeError(
            "missing package assets; "
            f"wheel_ui={missing_wheel}, source_ui={missing_source}, "
            f"wheel_benchmark={missing_wheel_benchmark}, "
            f"source_benchmark={missing_source_benchmark}, "
            f"source_checkout={missing_source_checkout}, "
            f"wheel_cache={unexpected_wheel_cache}, source_cache={unexpected_source_cache}"
        )
    print("Package contains all workbench and benchmark assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
