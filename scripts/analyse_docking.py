#!/usr/bin/env python3
"""Analyse docking poses with PoseBusters and OST.

The script processes one docking run directory (for example
``docking/gnina_redock``) and writes pose-level CSV and Parquet files beneath a
shared analysis directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem, RDLogger


SUPPORTED_METHODS = {"gnina", "smina", "diffdock"}


def file_ok(path: Path) -> bool:
    """Return whether a file exists and is non-empty."""
    return path.is_file() and path.stat().st_size > 0


def infer_similarity_label(path: Path) -> str:
    """Infer a date label from a similarity TSV filename."""
    match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    if match is None:
        raise ValueError(f"Could not infer date label from similarity TSV: {path}")
    return match.group(0)


def load_similarity_dicts(
    similarity_tsvs: Sequence[Path] | None,
) -> dict[str, dict[str, str]] | None:
    """Load optional similarity TSVs and suffix metric names by date."""
    if not similarity_tsvs:
        return None

    metric_cols = ["sucos_shape", "pocket_qcov", "sucos_shape_pocket_qcov"]
    similarity_dict: dict[str, dict[str, str]] = {}

    for similarity_tsv in similarity_tsvs:
        path = similarity_tsv.expanduser().resolve()
        if not file_ok(path):
            raise FileNotFoundError(f"Similarity TSV not found or empty: {path}")

        label = infer_similarity_label(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or reader.fieldnames[0] is None:
                raise ValueError(f"Similarity TSV has no header: {path}")
            missing = set(metric_cols) - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Similarity TSV is missing columns {sorted(missing)}: {path}"
                )

            id_column = reader.fieldnames[0]
            for row in reader:
                complex_id = row[id_column]
                similarity_dict.setdefault(complex_id, {})
                for col in metric_cols:
                    similarity_dict[complex_id][f"{col}-{label}"] = row[col]

    return similarity_dict


def add_similarity_metrics(
    df: pd.DataFrame,
    similarity_dict: dict[str, dict[str, str]] | None,
) -> pd.DataFrame:
    """Attach optional similarity metrics for each comparison complex."""
    if similarity_dict is None:
        return df
    metric_cols = sorted({col for values in similarity_dict.values() for col in values})
    for col in metric_cols:
        df[col] = df["compare_id"].map(
            lambda complex_id: similarity_dict.get(complex_id, {}).get(col)
        )
    return df


def structure_root_from_run(run_dir: Path) -> Path:
    """Resolve the dataset root from ``<dataset>/docking/<run>``."""
    if run_dir.parent.name != "docking":
        raise ValueError(
            f"Run directory must be directly beneath a 'docking' directory: {run_dir}"
        )
    structure_root = run_dir.parent.parent
    if not (structure_root / "prepared_inputs" / "redock").is_dir():
        raise FileNotFoundError(
            f"Redock prepared inputs not found beneath dataset root: {structure_root}"
        )
    return structure_root


def reference_paths(structure_root: Path, complex_id: str) -> tuple[Path, Path]:
    """Return the redock receptor and crystallographic ligand for one complex."""
    directory = structure_root / "prepared_inputs" / "redock" / complex_id
    protein = directory / f"{complex_id}_prepared.pdb"
    ligand = directory / f"{complex_id}_reference_ligand.sdf"
    if not file_ok(protein):
        raise FileNotFoundError(f"Missing reference receptor: {protein}")
    if not file_ok(ligand):
        raise FileNotFoundError(f"Missing reference ligand: {ligand}")
    return protein, ligand


def docked_protein_path(structure_root: Path, mode: str, complex_id: str) -> Path:
    """Return the exact receptor used for one docking mode."""
    directory = structure_root / "prepared_inputs" / mode / complex_id
    if not directory.is_dir():
        raise FileNotFoundError(f"Prepared receptor directory not found: {directory}")

    if mode == "redock":
        path = directory / f"{complex_id}_prepared.pdb"
    elif mode == "fragment_crossdock":
        matches = sorted(directory.glob("fragment_crossdock_*_aligned.pdb"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one fragment-crossdock receptor in {directory}; found {len(matches)}"
            )
        path = matches[0]
    else:
        path = directory / f"{mode}_prepared_aligned.pdb"

    if not file_ok(path):
        raise FileNotFoundError(f"Docked receptor not found or empty: {path}")
    return path


def seed_dirs(complex_dir: Path) -> list[Path]:
    """Return immediate seed directories for one docked complex."""
    directories = sorted(path for path in complex_dir.glob("seed_*") if path.is_dir())
    if not directories:
        raise FileNotFoundError(f"No seed directories found in: {complex_dir}")
    return directories


def pose_dir(seed_dir: Path) -> Path:
    """Return the canonical docked-pose directory."""
    directory = seed_dir / "docked_poses"
    if not directory.is_dir():
        raise FileNotFoundError(f"Docked-pose directory not found: {directory}")
    return directory


def format_chain_mapping(mapping: dict | None) -> str | None:
    """Format an OST chain mapping as a compact string."""
    if not mapping:
        return None
    return ",".join(f"{key}:{value}" for key, value in sorted(mapping.items()))


def ost_to_df(path: Path) -> pd.DataFrame:
    """Convert OST ligand-comparison output to a pose-level table."""
    with path.open("r", encoding="utf-8") as handle:
        ost = json.load(handle)

    lddt_rows = []
    for row in ost["lddt_pli"]["full_results"]:
        sample = int(row["model_ligand"].rsplit(":", 1)[1])
        score = row.get("score")
        lddt_rows.append(
            {"sample": sample, "lddt_pli": float(score) if score is not None else 0.0}
        )

    rmsd_rows = []
    for row in ost["rmsd"]["full_results"]:
        sample = int(row["model_ligand"].rsplit(":", 1)[1])
        score = row.get("score")
        rmsd_rows.append(
            {
                "sample": sample,
                "lig_rmsd": float(score) if score is not None else 99.0,
                "rmsd_chain_mapping": format_chain_mapping(row.get("chain_mapping")),
                "lddt_lp": row.get("lddt_lp"),
                "pocket_bb_rmsd": row.get("bb_rmsd"),
            }
        )
    return pd.DataFrame(lddt_rows).merge(
        pd.DataFrame(rmsd_rows), on="sample", how="outer"
    )


def posebusters_to_df(path: Path) -> pd.DataFrame:
    """Convert PoseBusters output to a pose-level validity table."""
    df = pd.read_csv(path).copy()

    if "position" in df.columns:
        df["sample"] = df["position"].astype(int)
    else:
        df["sample"] = range(len(df))

    meta_cols = {"file", "molecule", "position", "sample"}
    metric_cols = [col for col in df.columns if col not in meta_cols]

    if not metric_cols:
        raise ValueError(f"PoseBusters output contains no check columns: {path}")

    df["pb_valid"] = df[metric_cols].all(axis=1)

    return df[["sample", "pb_valid"]]


def sdf_to_pose_df(
    path: Path, score_property: str, flip_sign: bool = False
) -> pd.DataFrame:
    """Read pose scores from a ranked SDF file."""
    rows: list[dict[str, object]] = []
    for sample, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False)):
        if mol is None:
            raise ValueError(f"Invalid molecule at position {sample} in {path}")
        if not mol.HasProp(score_property):
            raise ValueError(f"Pose {sample} in {path} is missing '{score_property}'")
        score = float(mol.GetProp(score_property))
        rows.append({"sample": sample, "rank_score": -score if flip_sign else score})
    if not rows:
        raise ValueError(f"No poses found in SDF: {path}")
    return pd.DataFrame(rows)


def rank_molecules(
    molecules: list[Chem.Mol],
    score_property: str,
    descending: bool,
) -> list[Chem.Mol]:
    """Rank molecules by an SDF score property and annotate their rank."""
    for molecule in molecules:
        if not molecule.HasProp(score_property):
            raise ValueError(
                f"Molecule is missing required score property '{score_property}'"
            )
    ranked = sorted(
        molecules,
        key=lambda molecule: float(molecule.GetProp(score_property)),
        reverse=descending,
    )
    for rank, molecule in enumerate(ranked, start=1):
        molecule.SetIntProp("rank", rank)
    return ranked


def join_diffdock_sdfs(input_dir: Path, output_sdf: Path) -> None:
    """Combine DiffDock rank files while preserving confidence and rank metadata."""
    files = sorted(input_dir.glob("rank*_confidence*.sdf"))
    if not files:
        raise FileNotFoundError(f"No DiffDock SDF files found in: {input_dir}")

    def parse_rank_conf(path: Path) -> tuple[int, float]:
        match = re.search(r"rank(\d+)_confidence([-\d\.eE]+)", path.stem)
        if match is None:
            raise ValueError(f"Unexpected DiffDock filename: {path.name}")
        return int(match.group(1)), float(match.group(2))

    with Chem.SDWriter(str(output_sdf)) as writer:
        for path in sorted(files, key=lambda item: parse_rank_conf(item)[0]):
            molecules = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False)]
            if len(molecules) != 1 or molecules[0] is None:
                raise ValueError(f"Expected exactly one valid molecule in {path}")
            rank, confidence = parse_rank_conf(path)
            molecule = molecules[0]
            molecule.SetIntProp("rank", rank)
            molecule.SetProp("confidence", f"{confidence:.6f}")
            writer.write(molecule)


def join_docking_sdfs(
    input_dir: Path,
    output_sdf: Path,
    score_property: str,
    descending: bool,
) -> None:
    """Combine all GNINA/Smina pose files and rank poses globally."""
    files = sorted(input_dir.glob("*_docked_*.sdf"))
    if not files:
        raise FileNotFoundError(f"No docking SDF files found in: {input_dir}")

    molecules: list[Chem.Mol] = []
    for path in files:
        for index, molecule in enumerate(Chem.SDMolSupplier(str(path), removeHs=False)):
            if molecule is None:
                raise ValueError(f"Invalid molecule at position {index} in {path}")
            molecules.append(molecule)

    with Chem.SDWriter(str(output_sdf)) as writer:
        for molecule in rank_molecules(molecules, score_property, descending):
            writer.write(molecule)


def run_posebusters(
    buster: PoseBusters,
    sdf_path: Path,
    receptor_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    """Run PoseBusters unless a valid cached output is available."""
    if file_ok(output_path) and not overwrite:
        return
    molecules = list(Chem.SDMolSupplier(str(sdf_path), removeHs=False))
    if not molecules or any(molecule is None for molecule in molecules):
        raise ValueError(f"Could not read all poses from {sdf_path}")
    results = buster.bust(molecules, None, receptor_path)
    results.to_csv(output_path, index=False)


def run_ost(
    sdf_path: Path,
    docked_receptor_path: Path,
    reference_receptor_path: Path,
    reference_ligand_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    """Run OST ligand-structure comparison unless cached output is available."""
    if file_ok(output_path) and not overwrite:
        return
    command = [
        "ost",
        "compare-ligand-structures",
        "-m",
        str(docked_receptor_path),
        "-ml",
        str(sdf_path),
        "-r",
        str(reference_receptor_path),
        "-rl",
        str(reference_ligand_path),
        "--lddt-pli",
        "--rmsd",
        "-o",
        str(output_path),
        "-v",
        "0",
        "--full-results",
    ]
    subprocess.run(command, check=True)


def process_complex(
    complex_dir: Path,
    structure_root: Path,
    method: str,
    mode: str,
    run_label: str,
    overwrite: bool,
    similarity_dict: dict[str, dict[str, str]] | None,
) -> pd.DataFrame:
    """Analyse all seeds for one complex."""
    complex_id = complex_dir.name
    RDLogger.DisableLog("rdApp.warning")
    buster = PoseBusters(config="dock")

    reference_receptor, reference_ligand = reference_paths(structure_root, complex_id)
    docked_receptor = docked_protein_path(structure_root, mode, complex_id)
    reference_mol = Chem.MolFromMolFile(str(reference_ligand), removeHs=False)
    if reference_mol is None:
        raise ValueError(f"Could not read reference ligand: {reference_ligand}")
    ligand_smiles = Chem.MolToSmiles(reference_mol)

    if method == "gnina":
        score_property, flip_sign, descending = "CNNscore", False, True
    elif method == "smina":
        score_property, flip_sign, descending = "minimizedAffinity", True, False
    elif method == "diffdock":
        score_property, flip_sign, descending = "confidence", False, True
    else:
        raise ValueError(f"Unsupported method: {method}")

    results: list[pd.DataFrame] = []
    for seed_dir in seed_dirs(complex_dir):
        raw_pose_dir = pose_dir(seed_dir)
        joined_sdf = seed_dir / f"{complex_id}_{run_label}_joined.sdf"

        if method == "diffdock":
            if not file_ok(joined_sdf) or overwrite:
                join_diffdock_sdfs(raw_pose_dir, joined_sdf)
        else:
            if not file_ok(joined_sdf) or overwrite:
                join_docking_sdfs(raw_pose_dir, joined_sdf, score_property, descending)

        pb_path = seed_dir / f"PB_{complex_id}_{run_label}.csv"
        run_posebusters(buster, joined_sdf, docked_receptor, pb_path, overwrite)
        ost_path = seed_dir / f"OST_{complex_id}_{run_label}.json"
        run_ost(
            joined_sdf,
            docked_receptor,
            reference_receptor,
            reference_ligand,
            ost_path,
            overwrite,
        )

        pose_df = sdf_to_pose_df(joined_sdf, score_property, flip_sign=flip_sign)
        pb_df = posebusters_to_df(pb_path)
        ost_df = ost_to_df(ost_path)
        ost_df["compare_id"] = complex_id

        df = pose_df.merge(pb_df, on="sample", how="left", validate="one_to_one").merge(
            ost_df, on="sample", how="left", validate="one_to_one"
        )
        df = add_similarity_metrics(df, similarity_dict)
        df["dock_prot"] = mode
        df["complex_id"] = complex_id
        df["seed"] = seed_dir.name
        df["method"] = method
        df["ligand_smiles"] = ligand_smiles
        results.append(df)

    return pd.concat(results, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="One method/mode docking run directory."
    )
    parser.add_argument("--method", required=True, choices=sorted(SUPPORTED_METHODS))
    parser.add_argument("--mode", required=True, help="Prepared docking mode name.")
    parser.add_argument(
        "--similarity-tsv",
        type=Path,
        nargs="+",
        default=None,
        help="Optional similarity TSV file(s).",
    )
    parser.add_argument(
        "--num-workers",
        "-j",
        type=int,
        default=8,
        help="Number of complexes analysed in parallel (default: 8).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cached OST/PoseBusters outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis root (default: sibling analysis directory).",
    )
    parser.add_argument("--suffix", default="", help="Optional output filename suffix.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    """Run docking analysis."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("posebusters").setLevel(logging.ERROR)
    RDLogger.DisableLog("rdApp.warning")

    if shutil.which("ost") is None:
        raise FileNotFoundError("OST executable 'ost' was not found on PATH")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Docking run directory not found: {run_dir}")
    expected_name = f"{args.method}_{args.mode}"
    if run_dir.name != expected_name:
        raise ValueError(
            f"Run directory name must be '{expected_name}', got '{run_dir.name}'"
        )

    structure_root = structure_root_from_run(run_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir.parent.parent / "analysis"
    )
    similarity_dict = load_similarity_dicts(args.similarity_tsv)
    run_label = run_dir.name

    complex_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    if not complex_dirs:
        raise ValueError(f"No complex directories found in: {run_dir}")

    all_results: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    jobs = [
        (
            complex_dir,
            structure_root,
            args.method,
            args.mode,
            run_label,
            args.overwrite,
            similarity_dict,
        )
        for complex_dir in complex_dirs
    ]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_complex, *job): job[0].name for job in jobs}
        for future in as_completed(futures):
            complex_id = futures[future]
            try:
                all_results.append(future.result())
                logging.info("Finished %s", complex_id)
            except Exception as exc:
                failures.append((complex_id, str(exc)))
                logging.exception("Failed %s", complex_id)

    if not all_results:
        raise RuntimeError("Analysis produced no results")

    final_df = pd.concat(all_results, ignore_index=True)
    column_order = [
        "complex_id",
        "dock_prot",
        "method",
        "seed",
        "sample",
        "rank_score",
        "pb_valid",
        "lig_rmsd",
        "lddt_pli",
        "compare_id",
        "lddt_lp",
        "pocket_bb_rmsd",
        "rmsd_chain_mapping",
        "ligand_smiles",
    ]
    existing = [column for column in column_order if column in final_df.columns]
    extra = [column for column in final_df.columns if column not in column_order]
    final_df = final_df[existing + extra]

    suffix = args.suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    base_name = f"{run_label}{suffix}"
    csv_dir = output_dir / "csv"
    parquet_dir = output_dir / "parquet"
    csv_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(csv_dir / f"{base_name}.csv", index=False)
    final_df.to_parquet(parquet_dir / f"{base_name}.parquet", index=False)

    logging.info("Wrote %d pose rows for %s", len(final_df), run_label)
    if failures:
        failure_path = output_dir / "logs" / f"{run_label}_failed_complexes.tsv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", encoding="utf-8") as handle:
            handle.write("complex_id\treason\n")
            for complex_id, reason in failures:
                handle.write(
                    f"{complex_id}\t{reason.replace(chr(9), ' ').replace(chr(10), ' ')}\n"
                )
        raise SystemExit(f"Analysis completed with {len(failures)} failed complexes")


if __name__ == "__main__":
    main()
