#!/usr/bin/env python3
"""Generate ``annotated_complexes.csv`` from Fragalysis metadata and complexes."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem
from tqdm import tqdm


CODE_COL = "Code"
FRAGMENT_COL = "[Series] Fragment Screen"
ARTEFACT_COL = "[Other] Artefact"
OUTPUT_COLS = ["complex_name", "smiles", "fragment_screen", "pb_valid", "artefact"]


def as_bool(value: object) -> bool:
    """Parse common boolean values from Fragalysis metadata."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}


def read_smiles(ligand_path: Path) -> str:
    """Return canonical isomeric SMILES from the first ligand in an SDF file."""
    molecules = [mol for mol in Chem.SDMolSupplier(str(ligand_path), removeHs=False)]
    if not molecules or molecules[0] is None:
        raise ValueError(f"Could not read ligand: {ligand_path}")
    return Chem.MolToSmiles(molecules[0], canonical=True, isomericSmiles=True)


def check_posebusters(ligand_path: Path, protein_path: Path) -> bool:
    """Return whether all PoseBusters dock-mode checks pass."""
    results = PoseBusters(config="dock").bust(ligand_path, None, protein_path)
    checks = results.select_dtypes(include="bool")
    if checks.empty:
        raise ValueError(f"PoseBusters returned no boolean checks for {ligand_path}")
    return bool(checks.to_numpy().all())


def process_complex(task: tuple[str, str, object, object]) -> dict[str, object]:
    """Annotate one aligned complex."""
    complex_name, complexes_dir, fragment_screen, artefact = task
    complex_dir = Path(complexes_dir) / complex_name
    ligand_path = complex_dir / f"{complex_name}_ligand.sdf"
    protein_path = complex_dir / f"{complex_name}_delig-desolv.pdb"
    if not ligand_path.is_file():
        raise FileNotFoundError(f"Missing ligand: {ligand_path}")
    if not protein_path.is_file():
        raise FileNotFoundError(f"Missing protein: {protein_path}")

    return {
        "complex_name": complex_name,
        "smiles": read_smiles(ligand_path),
        "fragment_screen": as_bool(fragment_screen),
        "pb_valid": check_posebusters(ligand_path, protein_path),
        "artefact": as_bool(artefact),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_csv", type=Path, help="Fragalysis metadata.csv file.")
    parser.add_argument(
        "complexes_dir", type=Path, help="Fragalysis aligned_files directory."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: sibling annotated_complexes.csv).",
    )
    parser.add_argument(
        "--num-workers",
        "-j",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: available CPUs).",
    )
    return parser


def main() -> None:
    """Generate the standard OpenBind annotation table."""
    args = build_parser().parse_args()
    metadata_path = args.metadata_csv.expanduser().resolve()
    complexes_dir = args.complexes_dir.expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not complexes_dir.is_dir():
        raise FileNotFoundError(f"Complex directory not found: {complexes_dir}")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    metadata = pd.read_csv(metadata_path)
    required = {CODE_COL, FRAGMENT_COL, ARTEFACT_COL}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")
    if metadata[CODE_COL].duplicated().any():
        duplicates = metadata.loc[metadata[CODE_COL].duplicated(), CODE_COL].unique()
        raise ValueError(
            f"Duplicate values in '{CODE_COL}': {duplicates[:10].tolist()}"
        )

    metadata = metadata.copy()
    metadata[CODE_COL] = metadata[CODE_COL].astype(str).str.strip()
    metadata = metadata.set_index(CODE_COL)
    complex_names = sorted(
        path.name for path in complexes_dir.iterdir() if path.is_dir()
    )
    if not complex_names:
        raise ValueError(f"No complex directories found in: {complexes_dir}")

    missing_metadata = [name for name in complex_names if name not in metadata.index]
    if missing_metadata:
        raise ValueError(
            f"Complex directories missing from metadata.csv: {missing_metadata[:10]}"
        )

    tasks = [
        (
            complex_name,
            str(complexes_dir),
            metadata.at[complex_name, FRAGMENT_COL],
            metadata.at[complex_name, ARTEFACT_COL],
        )
        for complex_name in complex_names
    ]

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        rows = list(
            tqdm(
                executor.map(process_complex, tasks),
                total=len(tasks),
                desc="Annotating complexes",
                unit="complex",
            )
        )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else complexes_dir.parent / "annotated_complexes.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLS).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} complexes to {output_path}")


if __name__ == "__main__":
    main()
