#!/usr/bin/env python3
"""Submit docking-analysis jobs for all configured method/mode combinations."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ANALYSIS_SCRIPT = SCRIPT_DIR / "analyse_docking.py"
DEFAULT_SOFTWARE_SETTINGS = REPO_ROOT / "configs" / "docking" / "software_settings.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def configured_runs(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Return all method/mode combinations declared by the docking config."""
    modes = config.get("modes")
    methods = config.get("methods")
    if (
        not isinstance(modes, list)
        or not modes
        or not all(isinstance(x, str) for x in modes)
    ):
        raise ValueError("Docking config 'modes' must be a non-empty list of strings")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Docking config 'methods' must be a non-empty mapping")
    return [(str(method), mode) for method in methods for mode in modes]


def cpu_profile(settings: dict[str, Any]) -> dict[str, Any]:
    """Return the CPU Slurm profile and validate its worker count."""
    profiles = settings.get("profiles")
    if not isinstance(profiles, dict) or not isinstance(profiles.get("cpu"), dict):
        raise ValueError("Software settings must define profiles.cpu")
    profile = profiles["cpu"]
    cpus = profile.get("cpus-per-task")
    if not isinstance(cpus, int) or cpus < 1:
        raise ValueError("profiles.cpu.cpus-per-task must be a positive integer")
    return profile


def sbatch_command(
    profile: dict[str, Any],
    job_name: str,
    output_log: Path,
    wrap_command: list[str],
) -> list[str]:
    """Build one generic ``sbatch`` command from a profile mapping."""
    command = ["sbatch"]
    for key, value in profile.items():
        if value in (None, "", False):
            continue
        command.append(f"--{key}" if value is True else f"--{key}={value}")
    command += [
        f"--job-name={job_name}",
        f"--output={output_log}",
        f"--wrap={shlex.join(wrap_command)}",
    ]
    return command


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "docking_dir",
        type=Path,
        help="Docking output directory containing method_mode run directories.",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Docking experiment YAML file."
    )
    parser.add_argument(
        "--software-settings",
        type=Path,
        default=DEFAULT_SOFTWARE_SETTINGS,
        help=f"Software and Slurm YAML (default: {DEFAULT_SOFTWARE_SETTINGS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis root (default: sibling analysis directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting.",
    )
    return parser


def main() -> None:
    """Submit one analysis job for every available configured docking run."""
    args = build_parser().parse_args()

    if not ANALYSIS_SCRIPT.is_file():
        raise FileNotFoundError(f"Analysis script not found: {ANALYSIS_SCRIPT}")

    docking_dir = args.docking_dir.expanduser().resolve()
    if not docking_dir.is_dir():
        raise FileNotFoundError(f"Docking directory not found: {docking_dir}")

    config = load_yaml(args.config.expanduser().resolve())
    settings = load_yaml(args.software_settings.expanduser().resolve())

    profile = cpu_profile(settings)
    workers = int(profile["cpus-per-task"])

    analysis_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else docking_dir.parent / "analysis"
    )

    log_dir = analysis_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = configured_runs(config)

    available_runs = []
    missing_runs = []

    for method, mode in runs:
        run_dir = docking_dir / f"{method}_{mode}"
        if run_dir.is_dir():
            available_runs.append((method, mode, run_dir))
        else:
            missing_runs.append(run_dir)

    if missing_runs:
        print(
            "Warning: configured docking run directories are missing "
            "and will be skipped:",
            flush=True,
        )
        for run_dir in missing_runs:
            print(f"  {run_dir}", flush=True)

    if not available_runs:
        raise FileNotFoundError("None of the configured docking run directories exist.")

    submitted = 0
    failed = 0

    for method, mode, run_dir in available_runs:
        job_name = f"ana_{method}_{mode}"

        wrap = [
            "python",
            "-u",
            str(ANALYSIS_SCRIPT),
            str(run_dir),
            "--method",
            method,
            "--mode",
            mode,
            "--output-dir",
            str(analysis_dir),
            "--num-workers",
            str(workers),
        ]

        command = sbatch_command(
            profile=profile,
            job_name=job_name,
            output_log=log_dir / f"{job_name}-%j.out",
            wrap_command=wrap,
        )

        if args.dry_run:
            print(shlex.join(command))
            submitted += 1
            continue

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(
                f"Failed to submit {job_name}: {exc}",
                flush=True,
            )
        else:
            submitted += 1
            print(f"Submitted {job_name}", flush=True)

    print()

    if args.dry_run:
        print(f"Analysis jobs to submit: {submitted}")
    else:
        print(f"Successfully submitted analysis jobs: {submitted}")

    if failed:
        print(f"Failed submissions: {failed}")

    if missing_runs:
        print(f"Skipped missing runs: {len(missing_runs)}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
