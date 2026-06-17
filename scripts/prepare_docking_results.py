#!/usr/bin/env python3
"""Prepare the final OpenBind pose-level docking benchmark table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


FINAL_DATA_COLS = [
    "source",
    "method",
    "dock_prot",
    "complex_id",
    "seed",
    "sample",
    "rank",
    "rank_score",
    "rank_by_pair_iptm",
    "pair_iptm",
    "lddt_lp",
    "pocket_bb_rmsd",
    "lig_rmsd",
    "lddt_pli",
    "pb_valid",
    "ligand_smiles",
    "fragment_screen",
    "artefact",
    "pb_valid_groundtruth",
    "filtered",
    "pocket_qcov",
    "sucos_shape",
    "sucos_shape_pocket_qcov",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the top level."""
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def to_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean representations to ``bool``."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "t", "1", "yes", "y"})
    )


def normalise_seed_column(series: pd.Series) -> pd.Series:
    """Convert labels such as ``seed_123`` to integer seed values."""
    values = series.astype(str).str.extract(r"^seed_(\d+)$", expand=False)
    if values.isna().any():
        bad = series[values.isna()].astype(str).unique()[:10]
        raise ValueError(f"Invalid seed labels: {bad.tolist()}")
    return values.astype("Int64")


def normalise_sample_column(series: pd.Series) -> pd.Series:
    """Convert pose sample identifiers to integers."""
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        bad = series[values.isna()].astype(str).unique()[:10]
        raise ValueError(f"Invalid sample values: {bad.tolist()}")
    return values.astype("Int64")


def read_annotations(path: Path) -> pd.DataFrame:
    """Read and validate ``annotated_complexes.csv``."""
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    annotations = pd.read_csv(path)
    required = {"complex_name", "smiles", "fragment_screen", "pb_valid", "artefact"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotation file is missing columns: {sorted(missing)}")
    if annotations["complex_name"].duplicated().any():
        duplicates = annotations.loc[
            annotations["complex_name"].duplicated(), "complex_name"
        ].unique()
        raise ValueError(f"Duplicate complex_name entries: {duplicates[:10].tolist()}")

    annotations = annotations.rename(
        columns={
            "complex_name": "complex_id",
            "smiles": "ligand_smiles",
            "pb_valid": "pb_valid_groundtruth",
        }
    )
    annotations["fragment_screen"] = to_bool(annotations["fragment_screen"])
    annotations["pb_valid_groundtruth"] = to_bool(annotations["pb_valid_groundtruth"])
    annotations["artefact"] = to_bool(annotations["artefact"])
    annotations["filtered"] = (~annotations["pb_valid_groundtruth"]) | annotations[
        "artefact"
    ]
    return annotations[
        [
            "complex_id",
            "ligand_smiles",
            "fragment_screen",
            "artefact",
            "pb_valid_groundtruth",
            "filtered",
        ]
    ].copy()


def read_docking_file(path: Path) -> pd.DataFrame:
    """Read and standardise one analysed docking Parquet file."""
    df = pd.read_parquet(path).rename(
        columns={
            "lddt-pli": "lddt_pli",
            "pocket_qcov-2021-09-30": "pocket_qcov",
            "sucos_shape-2021-09-30": "sucos_shape",
            "sucos_shape_pocket_qcov-2021-09-30": "sucos_shape_pocket_qcov",
        }
    )
    required = {
        "method",
        "dock_prot",
        "complex_id",
        "seed",
        "sample",
        "rank_score",
        "lig_rmsd",
        "lddt_pli",
        "pb_valid",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["seed"] = normalise_seed_column(df["seed"])
    df["sample"] = normalise_sample_column(df["sample"])
    for column in ("rank_score", "lig_rmsd", "lddt_pli"):
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["pb_valid"] = to_bool(df["pb_valid"])
    df["source"] = "docking"

    keep = [
        "source",
        "method",
        "dock_prot",
        "complex_id",
        "seed",
        "sample",
        "rank_score",
        "lddt_lp",
        "pocket_bb_rmsd",
        "lig_rmsd",
        "lddt_pli",
        "pb_valid",
        "pocket_qcov",
        "sucos_shape",
        "sucos_shape_pocket_qcov",
    ]
    return df[[column for column in keep if column in df.columns]].copy()


def configured_runs(config: dict[str, Any]) -> list[str]:
    """Return all configured method/mode run labels."""
    modes = config.get("modes")
    methods = config.get("methods")

    if (
        not isinstance(modes, list)
        or not modes
        or not all(isinstance(mode, str) for mode in modes)
    ):
        raise ValueError("Docking config 'modes' must be a non-empty list of strings")

    if not isinstance(methods, dict) or not methods:
        raise ValueError("Docking config 'methods' must be a non-empty mapping")

    return [f"{method}_{mode}" for method in methods for mode in modes]


def load_docking_data(
    parquet_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load available Parquet files for configured docking runs."""
    available_files: list[Path] = []
    missing_files: list[Path] = []

    for run_label in configured_runs(config):
        path = parquet_dir / f"{run_label}.parquet"

        if path.is_file() and path.stat().st_size > 0:
            available_files.append(path)
        else:
            missing_files.append(path)

    if missing_files:
        print(
            "Warning: configured analysis files are missing and will be skipped:",
            flush=True,
        )
        for path in missing_files:
            print(f"  {path}", flush=True)

    if not available_files:
        raise FileNotFoundError(
            "None of the configured analysis Parquet files are available."
        )

    print(
        f"Loading {len(available_files)} of "
        f"{len(available_files) + len(missing_files)} configured analysis runs."
    )

    return pd.concat(
        [read_docking_file(path) for path in available_files],
        ignore_index=True,
        sort=False,
    )


def add_annotations(docking_df: pd.DataFrame, annotation_file: Path) -> pd.DataFrame:
    """Merge ground-truth annotations and require complete coverage."""
    annotations = read_annotations(annotation_file)
    merged = docking_df.merge(
        annotations, on="complex_id", how="left", validate="many_to_one"
    )
    missing = merged.loc[merged["ligand_smiles"].isna(), "complex_id"].unique()
    if len(missing):
        raise ValueError(
            f"Docking complexes missing from annotation file: {missing[:10].tolist()}"
        )
    return merged


def method_seeds(config: dict[str, Any], method: str) -> list[int]:
    """Return configured seeds for one docking method."""
    methods = config.get("methods")
    if not isinstance(methods, dict) or not isinstance(methods.get(method), dict):
        raise ValueError(f"Docking config is missing method '{method}'")
    seeds = methods[method].get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError(
            f"Method '{method}' must define a non-empty integer seeds list"
        )
    return seeds


def num_top_poses(config: dict[str, Any]) -> int:
    """Return the number of top poses retained per method and complex."""
    analysis = config.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("Docking config must define an 'analysis' mapping")
    top_poses = analysis.get("num_top_poses")
    if not isinstance(top_poses, int) or top_poses < 1:
        raise ValueError("analysis.num_top_poses must be a positive integer")
    return top_poses


def select_final_data(docking_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Retain the configured number of top poses per method and complex."""
    top_poses = num_top_poses(config)
    selected_frames: list[pd.DataFrame] = []

    for method in sorted(docking_df["method"].dropna().unique()):
        seeds = method_seeds(config, str(method))
        method_df = docking_df[docking_df["method"].eq(method)].copy()

        if method in {"gnina", "smina"}:
            selected = method_df[
                method_df["seed"].eq(seeds[0]) & method_df["sample"].lt(top_poses)
            ]
        elif method == "diffdock":
            if top_poses % len(seeds) != 0:
                raise ValueError(
                    f"analysis.num_top_poses ({top_poses}) must be divisible by the number "
                    f"of DiffDock seeds ({len(seeds)})"
                )
            poses_per_seed = top_poses // len(seeds)
            selected = method_df[
                method_df["seed"].isin(seeds) & method_df["sample"].lt(poses_per_seed)
            ]
        else:
            raise ValueError(f"Unsupported method in analysed data: {method}")
        selected_frames.append(selected)

    final_df = pd.concat(selected_frames, ignore_index=True)
    if final_df.empty:
        raise ValueError("No docking rows matched the configured top-pose selection")
    if final_df["rank_score"].isna().any():
        raise ValueError("Selected docking rows contain missing rank_score values")

    final_df = final_df.sort_values(
        ["method", "dock_prot", "complex_id", "rank_score", "seed", "sample"],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
    ).copy()
    final_df["rank"] = final_df.groupby(
        ["method", "dock_prot", "complex_id"]
    ).cumcount()
    final_df["rank_by_pair_iptm"] = pd.NA
    final_df["pair_iptm"] = pd.NA

    for column in FINAL_DATA_COLS:
        if column not in final_df.columns:
            final_df[column] = pd.NA

    return (
        final_df[FINAL_DATA_COLS]
        .sort_values(
            ["method", "dock_prot", "complex_id", "rank", "seed", "sample"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def print_summary(df: pd.DataFrame) -> None:
    """Print final pose-table coverage."""
    print(f"Docking rows: {len(df)}")
    print(
        df.groupby(["method", "dock_prot"])
        .agg(
            n_rows=("complex_id", "size"),
            n_complexes=("complex_id", "nunique"),
            n_seeds=("seed", "nunique"),
            min_sample=("sample", "min"),
            max_sample=("sample", "max"),
        )
        .reset_index()
        .to_string(index=False)
    )


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    """Write final docking CSV and Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final_docking_pose_data.csv"
    parquet_path = output_dir / "final_docking_pose_data.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {parquet_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis_dir",
        type=Path,
        help="Analysis directory containing the 'parquet' subdirectory.",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Docking experiment YAML file."
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="annotated_complexes.csv (default: sibling dataset file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Final output directory (default: analysis directory).",
    )
    return parser


def main() -> None:
    """Prepare the final pose-level docking table."""
    args = build_parser().parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    if not analysis_dir.is_dir():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    parquet_dir = analysis_dir / "parquet"
    if not parquet_dir.is_dir():
        raise FileNotFoundError(f"Analysis Parquet directory not found: {parquet_dir}")

    config = load_yaml(args.config.expanduser().resolve())
    annotation_file = (
        args.annotation_file.expanduser().resolve()
        if args.annotation_file is not None
        else analysis_dir.parent / "annotated_complexes.csv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_dir
    )

    docking_df = load_docking_data(parquet_dir, config)
    docking_df = add_annotations(docking_df, annotation_file)
    final_df = select_final_data(docking_df, config)
    write_outputs(final_df, output_dir)
    print_summary(final_df)


if __name__ == "__main__":
    main()
