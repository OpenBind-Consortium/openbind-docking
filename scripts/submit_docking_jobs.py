#!/usr/bin/env python3
"""Submit OpenBind docking jobs from an experiment YAML configuration.

The positional input is the ``prepared_inputs`` directory. Configured modes must
match child directory names exactly. Software paths and Slurm profiles are read
from ``configs/docking/software_settings.yaml`` by default.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import yaml


SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
{sbatch_lines}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err

{command}
"""

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DOCK_SCRIPT = SCRIPT_DIR / "dock_complexes.py"
DEFAULT_SOFTWARE_SETTINGS = REPO_ROOT / "configs" / "docking" / "software_settings.yaml"
SUPPORTED_METHODS = {"gnina", "smina", "diffdock"}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_repo_path(value: str) -> Path:
    """Resolve a software path relative to the repository root."""
    path = Path(value).expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_diffdock_config(executable: Path, num_modes: int) -> None:
    """Require DiffDock's YAML pose count to match the experiment config."""
    config_path = executable / "default_inference_args.yaml"
    config = load_yaml(config_path)
    samples_per_complex = config.get("samples_per_complex")
    if not isinstance(samples_per_complex, int) or samples_per_complex < 1:
        raise ValueError(
            "DiffDock default_inference_args.yaml must define a positive integer "
            f"samples_per_complex: {config_path}"
        )
    if samples_per_complex < num_modes:
        raise ValueError(
            "DiffDock loads samples_per_complex from default_inference_args.yaml after "
            "parsing command-line arguments, so it must match methods.diffdock.num_modes. "
            f"Set samples_per_complex: {num_modes} in {config_path} "
            f"(currently {samples_per_complex}). See software/README.md."
        )


def require_modes(config: dict[str, Any]) -> list[str]:
    """Return the configured docking modes."""
    modes = config.get("modes")
    if (
        not isinstance(modes, list)
        or not modes
        or not all(isinstance(x, str) for x in modes)
    ):
        raise ValueError(
            "Experiment config 'modes' must be a non-empty list of strings"
        )
    if len(modes) != len(set(modes)):
        raise ValueError("Experiment config contains duplicate docking modes")
    return modes


def require_methods(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return and validate experiment method settings."""
    methods = config.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Experiment config 'methods' must be a non-empty mapping")

    unknown = set(methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"Unsupported docking methods: {sorted(unknown)}")

    validated: dict[str, dict[str, Any]] = {}
    for name, settings in methods.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Method '{name}' settings must be a mapping")
        if (
            "seeds" not in settings
            or not isinstance(settings["seeds"], list)
            or not settings["seeds"]
        ):
            raise ValueError(f"Method '{name}' must define a non-empty seeds list")
        if not all(isinstance(seed, int) for seed in settings["seeds"]):
            raise ValueError(f"All seeds for '{name}' must be integers")

        required = {"num_modes"}
        if name in {"gnina", "smina"}:
            required |= {"num_inputs", "exhaustiveness", "box_size"}
        missing = required - set(settings)
        if missing:
            raise ValueError(f"Method '{name}' is missing settings: {sorted(missing)}")

        for key in {"num_modes", "num_inputs", "exhaustiveness"} & set(settings):
            if not isinstance(settings[key], int) or settings[key] < 1:
                raise ValueError(f"{name}.{key} must be a positive integer")
        validated[name] = settings

    return validated


def require_software_methods(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return software method definitions from the software settings file."""
    methods = settings.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("Software settings must contain a 'methods' mapping")
    return methods


def require_profiles(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return Slurm profiles from the software settings file."""
    profiles = settings.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Software settings must contain a 'profiles' mapping")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Slurm profile '{name}' must be a mapping")
        if "cpus-per-task" not in profile:
            raise ValueError(f"Slurm profile '{name}' must define cpus-per-task")
    return profiles


def validate_centre(value: object) -> tuple[float, float, float] | None:
    """Validate an optional docking-box centre from the experiment config."""
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(
            "Experiment config 'centre' must contain exactly three numbers"
        )
    if not all(isinstance(item, (int, float)) for item in value):
        raise ValueError("All values in experiment config 'centre' must be numeric")
    return float(value[0]), float(value[1]), float(value[2])


def render_sbatch_lines(profile: dict[str, Any]) -> str:
    """Render Slurm profile entries as SBATCH directives."""
    lines: list[str] = []
    for key, value in profile.items():
        if value in (None, "", False):
            continue
        if value is True:
            lines.append(f"#SBATCH --{key}")
        else:
            lines.append(f"#SBATCH --{key}={value}")
    return "\n".join(lines)


def resolve_software_method(
    method_name: str,
    software_methods: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> tuple[Path, Path | None, str, dict[str, Any]]:
    """Resolve executable/container paths and the Slurm profile for one method."""
    settings = software_methods.get(method_name)
    if not isinstance(settings, dict):
        raise ValueError(f"Software settings are missing method '{method_name}'")

    executable_value = settings.get("executable")
    profile_name = settings.get("profile")
    if not isinstance(executable_value, str) or not executable_value:
        raise ValueError(f"Software method '{method_name}' must define executable")
    if not isinstance(profile_name, str) or profile_name not in profiles:
        raise ValueError(
            f"Software method '{method_name}' references unknown profile "
            f"'{profile_name}'"
        )

    executable = resolve_repo_path(executable_value)
    if method_name == "diffdock":
        if not executable.is_dir() or not (executable / "inference.py").is_file():
            raise FileNotFoundError(
                f"DiffDock directory or inference.py not found: {executable}"
            )
    elif not executable.is_file():
        raise FileNotFoundError(
            f"Executable for '{method_name}' not found: {executable}"
        )

    singularity: Path | None = None
    if settings.get("singularity"):
        singularity = resolve_repo_path(str(settings["singularity"]))
        if not singularity.is_file():
            raise FileNotFoundError(
                f"Singularity image for '{method_name}' not found: {singularity}"
            )

    return executable, singularity, profile_name, profiles[profile_name]


def build_docking_command(
    mode_dir: Path,
    output_dir: Path,
    method_name: str,
    method_cfg: dict[str, Any],
    executable: Path,
    mode: str,
    seed: int,
    centre: tuple[float, float, float] | None,
    overwrite: bool,
) -> list[str]:
    """Build one ``dock_complexes.py`` command."""
    label = method_name
    command = [
        "python",
        "-u",
        str(DOCK_SCRIPT),
        str(mode_dir),
        "--engine",
        method_name,
        "--method",
        str(executable),
        "--method-label",
        label,
        "--mode-label",
        mode,
        "--output-dir",
        str(output_dir),
        "--num-modes",
        str(method_cfg["num_modes"]),
        "--seed",
        str(seed),
    ]

    if method_name in {"gnina", "smina"}:
        command += [
            "--num-inputs",
            str(method_cfg["num_inputs"]),
            "--exhaustiveness",
            str(method_cfg["exhaustiveness"]),
            "--size",
            str(method_cfg["box_size"]),
        ]
        if method_cfg.get("scoring") is not None:
            command += ["--scoring", str(method_cfg["scoring"])]
        if method_name == "gnina" and method_cfg.get("cnn_model") not in (
            None,
            "default",
        ):
            command += [
                "--cnn-model",
                str(resolve_repo_path(str(method_cfg["cnn_model"]))),
            ]
        if centre is not None:
            command += [
                "--x-coord",
                str(centre[0]),
                "--y-coord",
                str(centre[1]),
                "--z-coord",
                str(centre[2]),
            ]

    if overwrite:
        command.append("--overwrite")
    return command


def wrap_singularity(
    command: list[str],
    image: Path | None,
    use_gpu: bool,
    prepared_inputs_dir: Path,
    mode_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Wrap a docking command in Singularity when an image is configured."""
    if image is None:
        return command

    bind_paths = {
        REPO_ROOT.resolve(),
        prepared_inputs_dir.resolve(),
        mode_dir.resolve(),
        output_dir.resolve(),
    }
    wrapped = ["singularity", "exec"]
    if use_gpu:
        wrapped.append("--nv")
    wrapped += [
        "--bind",
        ",".join(f"{path}:{path}" for path in sorted(bind_paths, key=str)),
        "--pwd",
        str(REPO_ROOT),
        str(image),
        *command,
    ]
    return wrapped


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prepared_inputs_dir",
        type=Path,
        help="Directory containing one prepared-input subdirectory per docking mode.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Docking experiment YAML file.",
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
        help="Docking output directory (default: sibling 'docking' directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print jobs without submitting them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing docking outputs.",
    )
    return parser


def main() -> None:
    """Submit configured docking jobs."""
    args = build_parser().parse_args()
    if not DOCK_SCRIPT.is_file():
        raise FileNotFoundError(f"Docking runner not found: {DOCK_SCRIPT}")

    prepared_inputs_dir = args.prepared_inputs_dir.expanduser().resolve()
    if not prepared_inputs_dir.is_dir():
        raise FileNotFoundError(
            f"Prepared-input directory not found: {prepared_inputs_dir}"
        )

    config_path = args.config.expanduser().resolve()
    software_path = args.software_settings.expanduser().resolve()
    experiment_cfg = load_yaml(config_path)
    software_cfg = load_yaml(software_path)

    modes = require_modes(experiment_cfg)
    methods = require_methods(experiment_cfg)
    software_methods = require_software_methods(software_cfg)
    profiles = require_profiles(software_cfg)
    centre = validate_centre(experiment_cfg.get("centre"))

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else prepared_inputs_dir.parent / "docking"
    )

    # Preflight all configured modes and software before submitting any jobs.
    mode_dirs: dict[str, Path] = {}
    for mode in modes:
        mode_dir = prepared_inputs_dir / mode
        if not mode_dir.is_dir():
            raise FileNotFoundError(f"Prepared mode directory not found: {mode_dir}")
        mode_dirs[mode] = mode_dir

    resolved_methods: dict[str, tuple[Path, Path | None, str, dict[str, Any]]] = {}
    for method_name, method_cfg in methods.items():
        resolved = resolve_software_method(method_name, software_methods, profiles)
        executable, _, _, _ = resolved
        if method_name == "diffdock":
            validate_diffdock_config(executable, method_cfg["num_modes"])
        resolved_methods[method_name] = resolved

    jobs = 0

    for mode in modes:
        mode_dir = mode_dirs[mode]

        for method_name, method_cfg in methods.items():
            executable, image, profile_name, profile = resolved_methods[method_name]

            use_gpu = bool(profile.get("gres"))
            label = method_name

            for seed in method_cfg["seeds"]:
                setup_label = f"{label}_{mode}"
                job_name = f"{setup_label}_seed{seed}"
                job_dir = output_dir / "slurm_jobs" / setup_label
                log_dir = output_dir / "slurm_logs" / setup_label

                command = build_docking_command(
                    mode_dir=mode_dir,
                    output_dir=output_dir,
                    method_name=method_name,
                    method_cfg=method_cfg,
                    executable=executable,
                    mode=mode,
                    seed=seed,
                    centre=centre,
                    overwrite=args.overwrite,
                )
                command = wrap_singularity(
                    command=command,
                    image=image,
                    use_gpu=use_gpu,
                    prepared_inputs_dir=prepared_inputs_dir,
                    mode_dir=mode_dir,
                    output_dir=output_dir,
                )

                script = SBATCH_TEMPLATE.format(
                    job_name=job_name,
                    sbatch_lines=render_sbatch_lines(profile),
                    log_dir=log_dir.resolve(),
                    command=shlex.join(command),
                )
                script_path = job_dir / f"{job_name}.sbatch"

                if args.dry_run:
                    print(f"[dry-run] {profile_name}: sbatch {script_path}")
                    print(textwrap.dedent(script))
                else:
                    job_dir.mkdir(parents=True, exist_ok=True)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    script_path.write_text(textwrap.dedent(script), encoding="utf-8")
                    subprocess.run(["sbatch", str(script_path)], check=True)
                    print(f"Submitted {job_name}")
                jobs += 1

    print(f"Total jobs: {jobs}")


if __name__ == "__main__":
    main()
