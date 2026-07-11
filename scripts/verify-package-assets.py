from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

UI_ASSETS = ("index.html", "app.css", "app.js")


def main() -> int:
    distribution_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = sorted(distribution_dir.glob("*.whl"))
    source_archives = sorted(distribution_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(source_archives) != 1:
        raise RuntimeError("expected exactly one wheel and one source archive")

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = set(archive.namelist())
    missing_wheel = [
        name for name in UI_ASSETS if f"inverse_agent/ui/{name}" not in wheel_names
    ]

    with tarfile.open(source_archives[0], "r:gz") as archive:
        source_names = {member.name for member in archive.getmembers()}
    missing_source = [
        name
        for name in UI_ASSETS
        if not any(path.endswith(f"/src/inverse_agent/ui/{name}") for path in source_names)
    ]

    if missing_wheel or missing_source:
        raise RuntimeError(
            f"missing UI assets; wheel={missing_wheel}, source={missing_source}"
        )
    print("Package contains all workbench assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
