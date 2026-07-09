"""Command line interface for Inverse-Agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from inverse_agent.adapters.registry import detect_workspace
from inverse_agent.eval import json_default
from inverse_agent.models import AutonomyLevel, Domain, RunSpec
from inverse_agent.workflow import run_django_replay


def profile_command(args: argparse.Namespace) -> int:
    profile = detect_workspace(Path(args.workspace))
    print(json.dumps(asdict(profile), default=json_default, indent=2))
    return 0


def run_django_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output).resolve()
    spec = RunSpec(
        goal=args.goal,
        workspace=workspace,
        domain=Domain.DJANGO,
        autonomy_level=AutonomyLevel.ASSISTED,
    )
    result = run_django_replay(spec, output_dir)
    print(json.dumps(asdict(result.trace), default=json_default, indent=2))
    return 0 if result.trace.status.value == "succeeded" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inverse-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="Detect a workspace profile")
    profile.add_argument("workspace")
    profile.set_defaults(func=profile_command)

    django = sub.add_parser("run-django", help="Run the Django replay workflow")
    django.add_argument("workspace")
    django.add_argument("--goal", default="Run Django checks and tests")
    django.add_argument("--output", default="runs")
    django.set_defaults(func=run_django_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
