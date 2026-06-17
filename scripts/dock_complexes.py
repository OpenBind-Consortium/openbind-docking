#!/usr/bin/env python3
"""Run one docking method over a prepared OpenBind docking-mode directory."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from datetime import datetime

Coord3D = tuple[float, float, float]
SUPPORTED_ENGINES = {"gnina", "smina", "diffdock"}


def log(message: str) -> None:
    """Print a timestamped status message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def validate_file(path: Path, label: str) -> Path:
    """Require a non-empty input file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty {label}: {path}")
    return path


def compute_centroid(sdf_path: Path) -> Coord3D:
    """Return the centroid of the first conformer in an SDF file."""
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"RDKit could not read a 3D ligand from: {sdf_path}")
    centroid = rdMolTransforms.ComputeCentroid(mol.GetConformer(0))
    return float(centroid.x), float(centroid.y), float(centroid.z)


def reference_complex_dir(mode_dir: Path, complex_id: str) -> Path:
    """Return the redock directory providing ligand inputs and reference poses."""
    root = mode_dir if mode_dir.name == "redock" else mode_dir.parent / "redock"
    complex_dir = root / complex_id
    if not complex_dir.is_dir():
        raise FileNotFoundError(f"Redock complex directory not found: {complex_dir}")
    return complex_dir


def receptor_path(
    complex_dir: Path,
    complex_id: str,
    mode_label: str,
) -> Path:
    """Resolve the receptor file using the fixed prepared-input naming scheme."""
    if mode_label == "redock":
        return validate_file(complex_dir / f"{complex_id}_prepared.pdb", "receptor")
    if mode_label == "fragment_crossdock":
        matches = sorted(complex_dir.glob("fragment_crossdock_*_aligned.pdb"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one fragment-crossdock receptor in {complex_dir}; "
                f"found {len(matches)}"
            )
        return validate_file(matches[0], "receptor")
    return validate_file(
        complex_dir / f"{mode_label}_prepared_aligned.pdb",
        "receptor",
    )


def ligand_input_path(reference_dir: Path, complex_id: str, input_num: int) -> Path:
    """Return one prepared ligand conformer."""
    return validate_file(
        reference_dir / f"{complex_id}_ligand_prepared_{input_num}.sdf",
        "ligand input",
    )


def reference_ligand_path(reference_dir: Path, complex_id: str) -> Path:
    """Return the crystallographic ligand used for docking-box placement."""
    return validate_file(
        reference_dir / f"{complex_id}_reference_ligand.sdf",
        "reference ligand",
    )


def build_box_args(
    size: str,
    reference_ligand: Path,
    centre: Coord3D | None,
) -> list[str]:
    """Build GNINA/Smina box arguments."""
    if size.startswith("autobox"):
        parts = size.split("_", maxsplit=1)
        args = ["--autobox_ligand", str(reference_ligand)]
        if len(parts) == 2:
            try:
                float(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid autobox size: {size}") from exc
            args += ["--autobox_add", parts[1]]
        return args

    try:
        box_size = float(size)
    except ValueError as exc:
        raise ValueError(
            "--size must be numeric or use the form autobox[_ADD]"
        ) from exc
    if box_size <= 0:
        raise ValueError("--size must be greater than zero")

    x, y, z = centre if centre is not None else compute_centroid(reference_ligand)
    return [
        "--center_x",
        str(x),
        "--center_y",
        str(y),
        "--center_z",
        str(z),
        "--size_x",
        str(box_size),
        "--size_y",
        str(box_size),
        "--size_z",
        str(box_size),
    ]


def add_cnn_model_args(command: list[str], cnn_model: str | None) -> None:
    """Append GNINA CNN model arguments."""
    if cnn_model in (None, "default"):
        return

    path = Path(cnn_model).expanduser().resolve()
    if path.is_file():
        command += ["--cnn_model", str(path)]
        return
    if path.is_dir():
        models = sorted(path.glob("*.pt"))
        if not models:
            raise FileNotFoundError(f"No .pt models found in: {path}")
        command.extend(["--cnn_model", *map(str, models)])
        return
    raise FileNotFoundError(f"CNN model path not found: {path}")


def diffdock_complete(poses_dir: Path, num_modes: int) -> bool:
    """Return whether a DiffDock run already contains the requested pose count."""
    poses = sorted(poses_dir.glob("rank*_confidence*.sdf"))
    return len(poses) >= num_modes and all(path.stat().st_size > 0 for path in poses)


def run_docking(
    complex_dir: Path,
    mode_dir: Path,
    complex_id: str,
    engine: str,
    method: Path,
    method_label: str,
    mode_label: str,
    output_dir: Path,
    input_num: int,
    num_modes: int,
    exhaustiveness: int,
    size: str,
    seed: int,
    scoring: str | None,
    cnn_model: str | None,
    centre: Coord3D | None,
    overwrite: bool,
) -> tuple[int, Path, Path]:
    """Run one docking task and return the exit code and output paths."""
    receptor = receptor_path(complex_dir, complex_id, mode_label)
    reference_dir = reference_complex_dir(mode_dir, complex_id)
    ligand_input = ligand_input_path(reference_dir, complex_id, input_num)
    reference_ligand = reference_ligand_path(reference_dir, complex_id)

    setup_label = f"{method_label}_{mode_label}"
    seed_dir = output_dir / setup_label / complex_id / f"seed_{seed}"
    poses_dir = seed_dir / "docked_poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    log_file = seed_dir / f"{complex_id}_{input_num}.log"

    if engine == "diffdock":
        if not overwrite and diffdock_complete(poses_dir, num_modes):
            return 0, poses_dir, log_file
        command = [
            sys.executable,
            "inference.py",
            "--config",
            "default_inference_args.yaml",
            "--protein_path",
            str(receptor),
            "--ligand_description",
            str(ligand_input),
            "--out_dir",
            str(seed_dir),
            "--complex_name",
            poses_dir.name,
        ]
        cwd = method
        expected_output: Path | None = None
    else:
        expected_output = poses_dir / f"{complex_id}_docked_{input_num}.sdf"
        if (
            not overwrite
            and expected_output.is_file()
            and expected_output.stat().st_size > 0
        ):
            return 0, poses_dir, log_file
        command = [
            str(method),
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand_input),
            "--out",
            str(expected_output),
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
            "--seed",
            str(seed),
            *build_box_args(size, reference_ligand, centre),
        ]
        if scoring is not None:
            command += ["--scoring", scoring]
        if engine == "gnina":
            add_cnn_model_args(command, cnn_model)
        cwd = None

    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            check=False,
        )

    if result.returncode == 0 and expected_output is not None:
        validate_file(expected_output, "docking output")
    return result.returncode, poses_dir, log_file


def parse_centre(args: argparse.Namespace) -> Coord3D | None:
    """Parse an optional explicit docking-box centre."""
    values = (args.x_coord, args.y_coord, args.z_coord)
    if any(value is not None for value in values) and not all(
        value is not None for value in values
    ):
        raise ValueError("Provide all of --x-coord, --y-coord and --z-coord, or none")
    if values[0] is None:
        return None
    return float(values[0]), float(values[1]), float(values[2])


def find_complexes(mode_dir: Path, complex_id: str | None) -> list[Path]:
    """Return selected prepared complex directories."""
    if complex_id is not None:
        path = mode_dir / complex_id
        if not path.is_dir():
            raise FileNotFoundError(f"Complex directory not found: {path}")
        return [path]
    complexes = sorted(path for path in mode_dir.iterdir() if path.is_dir())
    if not complexes:
        raise ValueError(f"No complex directories found in {mode_dir}")
    return complexes


def validate_method(engine: str, method: Path) -> Path:
    """Validate the configured docking executable or DiffDock directory."""
    path = method.expanduser().resolve()
    if engine == "diffdock":
        if not path.is_dir() or not (path / "inference.py").is_file():
            raise FileNotFoundError(
                f"DiffDock directory or inference.py not found: {path}"
            )
    elif not path.is_file():
        raise FileNotFoundError(f"Docking executable not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode_dir", type=Path, help="Prepared directory for one docking mode."
    )
    parser.add_argument("--engine", required=True, choices=sorted(SUPPORTED_ENGINES))
    parser.add_argument(
        "--method",
        required=True,
        type=Path,
        help="Executable or DiffDock directory.",
    )
    parser.add_argument(
        "--method-label", required=True, help="Method label used in output paths."
    )
    parser.add_argument(
        "--mode-label", required=True, help="Prepared mode directory name."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Docking output root."
    )
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--num-inputs", type=int, default=1)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--size", default="25", help="Box size or autobox[_ADD].")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scoring", default=None)
    parser.add_argument("--cnn-model", default=None)
    parser.add_argument("--complex-id", default=None, help="Run one complex only.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--x-coord", type=float, default=None)
    parser.add_argument("--y-coord", type=float, default=None)
    parser.add_argument("--z-coord", type=float, default=None)
    return parser


def main() -> None:
    """Run docking across the selected complexes."""
    args = build_parser().parse_args()
    mode_dir = args.mode_dir.expanduser().resolve()
    if not mode_dir.is_dir():
        raise FileNotFoundError(f"Prepared mode directory not found: {mode_dir}")
    if args.mode_label != mode_dir.name:
        raise ValueError(
            f"--mode-label must match the prepared directory name exactly: {mode_dir.name}"
        )
    if args.num_inputs < 1 or args.num_modes < 1 or args.exhaustiveness < 1:
        raise ValueError(
            "--num-inputs, --num-modes and --exhaustiveness must be positive"
        )

    method = validate_method(args.engine, args.method)

    output_dir = args.output_dir.expanduser().resolve()
    centre = parse_centre(args)
    complexes = find_complexes(mode_dir, args.complex_id)
    num_inputs = 1 if args.engine == "diffdock" else args.num_inputs

    log(f"Prepared mode: {mode_dir}")
    log(f"Output root: {output_dir}")
    log(f"Method: {args.method_label} ({args.engine})")
    log(f"Complexes: {len(complexes)}")

    start = time.time()
    failures: list[tuple[str, str, Path | None]] = []
    succeeded = 0
    attempted = 0

    for complex_dir in complexes:
        complex_id = complex_dir.name
        if args.overwrite:
            seed_dir = (
                output_dir
                / f"{args.method_label}_{args.mode_label}"
                / complex_id
                / f"seed_{args.seed}"
            )
            if seed_dir.exists():
                shutil.rmtree(seed_dir)

        for input_num in range(1, num_inputs + 1):
            attempted += 1
            task_id = f"{complex_id}_{input_num}"
            log(f"-> {task_id} ...")
            try:
                return_code, poses_dir, log_file = run_docking(
                    complex_dir=complex_dir,
                    mode_dir=mode_dir,
                    complex_id=complex_id,
                    engine=args.engine,
                    method=method,
                    method_label=args.method_label,
                    mode_label=args.mode_label,
                    output_dir=output_dir,
                    input_num=input_num,
                    num_modes=args.num_modes,
                    exhaustiveness=args.exhaustiveness,
                    size=str(args.size),
                    seed=args.seed,
                    scoring=args.scoring,
                    cnn_model=args.cnn_model,
                    centre=centre,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                log(f" ERROR: {exc}")
                failures.append((task_id, str(exc), None))
                continue

            if return_code == 0:
                succeeded += 1
                log(" OK")
            else:
                log(f" FAILED (exit {return_code})")
                failures.append((task_id, f"exit code {return_code}", log_file))

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    success_pct = 100.0 * succeeded / attempted if attempted else 0.0
    log(
        f"Summary: complexes={len(complexes)}, tasks={attempted}, "
        f"succeeded={succeeded} ({success_pct:.1f}%), failed={len(failures)}"
    )
    log(f"Elapsed time: {elapsed}")

    if failures:
        for task_id, error, log_file in failures:
            log_text = f"; log={log_file}" if log_file is not None else ""
            log(f" - {task_id}: {error}{log_text}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
