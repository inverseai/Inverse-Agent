from __future__ import annotations

import argparse

# torch is intentionally mentioned for adapter detection; the fixture does not
# import it so tests run without a GPU or heavyweight dependencies.


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print("torch smoke training complete" if args.smoke else "torch training complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

