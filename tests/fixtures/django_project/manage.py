#!/usr/bin/env python
from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if args == ["check"]:
        print("System check identified no issues (0 silenced).")
        return 0
    if args == ["test"]:
        print("Ran 1 test in 0.001s")
        print("OK")
        return 0
    if args == ["makemigrations", "--check", "--dry-run"]:
        print("No changes detected")
        return 0
    if args == ["migrate", "--plan"]:
        print("Planned operations: none")
        return 0
    print(f"unsupported fixture manage.py command: {args}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

