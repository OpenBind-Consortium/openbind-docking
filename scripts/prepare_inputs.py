#!/usr/bin/env python3
"""Prepare receptor and ligand inputs for the OpenBind docking workflow.

The script supports three preparation modes configured by YAML:

- ``redock`` prepares cognate receptors and ligand conformers.
- ``crossdock`` prepares one reference receptor and aligns it into each redock frame.
- ``fragment_crossdock`` selects the most similar fragment receptor and aligns it
  into each redock frame.

Cross-docking modes require redock outputs at
``<dataset_root>/prepared_inputs/redock``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import yaml
from openmm import unit
from openmm.app import ForceField, PDBFile
from pdbfixer import PDBFixer
from prody import confProDy, matchAlign, parsePDB, writePDB
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from tqdm import tqdm


ChainPolicy = Literal["single", "nearby", "all"]
MinimiseMode = Literal["none", "uff", "mmff94"]
DuplicateMode = Literal["none", "prefix", "smiles", "both"]
WorkerResult = tuple[str, bool, str | None]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_ROOT = REPO_ROOT / "configs" / "prepare_inputs"

RECEPTOR_PREPARED_NAME = "{complex_id}_prepared.pdb"
LIGAND_H_NAME = "{complex_id}_ligand_H.sdf"
LIGAND_CONFORMER_NAME = "{complex_id}_ligand_prepared_{i}.sdf"
REQUIRED_MODES = {"redock", "crossdock", "fragment_crossdock"}


def resolve_config_path(config_arg: str) -> Path:
    """Resolve an explicit YAML path or an exact built-in profile name."""
    candidate = Path(config_arg).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    builtin = CONFIG_ROOT / f"{config_arg}.yaml"
    if builtin.is_file():
        return builtin.resolve()

    raise FileNotFoundError(
        f"Could not resolve config '{config_arg}'. Expected a YAML file or one of "
        f"the built-in profiles in {CONFIG_ROOT}."
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a preparation YAML file and require a mapping at the top level."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the settings required by the selected preparation mode."""
    required = {"schema_version", "mode"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing required keys: {sorted(missing)}")

    if config["schema_version"] != 1:
        raise ValueError("Only preparation config schema_version 1 is supported")

    mode = config["mode"]
    if mode not in REQUIRED_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(REQUIRED_MODES))}")

    if mode == "redock":
        protein_cfg = config.get("protein")
        ligand_cfg = config.get("ligand")
        if not isinstance(protein_cfg, dict) or not isinstance(ligand_cfg, dict):
            raise ValueError("redock config requires 'protein' and 'ligand' mappings")

        required_protein = {
            "chain_policy",
            "ligand_cutoff_angstrom",
            "pH",
            "forcefield",
        }
        required_ligand = {"protonate", "minimise_conformers", "num_conformers", "seed"}
        missing_protein = required_protein - set(protein_cfg)
        missing_ligand = required_ligand - set(ligand_cfg)
        if missing_protein:
            raise ValueError(
                f"protein is missing required keys: {sorted(missing_protein)}"
            )
        if missing_ligand:
            raise ValueError(
                f"ligand is missing required keys: {sorted(missing_ligand)}"
            )

        if protein_cfg["chain_policy"] not in {"single", "nearby", "all"}:
            raise ValueError("protein.chain_policy must be one of: single, nearby, all")
        if ligand_cfg["minimise_conformers"] not in {"none", "uff", "mmff94"}:
            raise ValueError(
                "ligand.minimise_conformers must be one of: none, uff, mmff94"
            )
        if (
            not isinstance(ligand_cfg["num_conformers"], int)
            or ligand_cfg["num_conformers"] < 1
        ):
            raise ValueError("ligand.num_conformers must be a positive integer")
        if not isinstance(ligand_cfg["seed"], int):
            raise ValueError("ligand.seed must be an integer")
        ForceField(str(protein_cfg["forcefield"]))

    elif mode == "crossdock":
        protein_cfg = config.get("protein")
        if not isinstance(protein_cfg, dict):
            raise ValueError("crossdock config requires a 'protein' mapping")
        missing_protein = {"pH", "forcefield"} - set(protein_cfg)
        if missing_protein:
            raise ValueError(
                f"protein is missing required keys: {sorted(missing_protein)}"
            )
        ForceField(str(protein_cfg["forcefield"]))

    else:
        fragment_cfg = config.get("fragment_selection")
        if not isinstance(fragment_cfg, dict):
            raise ValueError("fragment_crossdock config requires 'fragment_selection'")
        required_fragment = {
            "csv_path",
            "pb_valid_filter",
            "artefact_filter",
            "duplicate_filter_mode",
        }
        missing_fragment = required_fragment - set(fragment_cfg)
        if missing_fragment:
            raise ValueError(
                "fragment_selection is missing required keys: "
                f"{sorted(missing_fragment)}"
            )
        if fragment_cfg["duplicate_filter_mode"] not in {
            "none",
            "prefix",
            "smiles",
            "both",
        }:
            raise ValueError(
                "fragment_selection.duplicate_filter_mode must be one of: "
                "none, prefix, smiles, both"
            )


def resolve_output_root(
    input_root: Path,
    config: dict[str, Any],
    output_dir: Path | None,
    crossdock_pdb: Path | None,
) -> tuple[Path, Path | None, str | None]:
    """Resolve the output directory and optional cross-docking reference receptor."""
    mode = str(config["mode"])
    reference_path: Path | None = None
    reference_id: str | None = None

    if mode == "crossdock":
        if crossdock_pdb is None:
            raise ValueError("crossdock mode requires --crossdock-pdb")
        reference_path = crossdock_pdb.expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Cross-docking reference PDB not found: {reference_path}"
            )
        reference_id = reference_path.stem
        default_name = reference_id
    else:
        if crossdock_pdb is not None:
            raise ValueError("--crossdock-pdb is only valid with the crossdock profile")
        default_name = mode

    if output_dir is None:
        output_root = input_root.parent / "prepared_inputs" / default_name
    else:
        output_root = output_dir.expanduser().resolve()

    return output_root, reference_path, reference_id


def write_runtime_config(
    config: dict[str, Any],
    input_root: Path,
    output_root: Path,
    config_path: Path,
    reference_path: Path | None,
    reference_id: str | None,
) -> None:
    """Write the exact configuration and resolved runtime paths used for a run."""
    snapshot = dict(config)
    snapshot["runtime"] = {
        "config_path": str(config_path),
        "input_root": str(input_root),
        "output_root": str(output_root),
    }
    if reference_path is not None:
        snapshot["runtime"]["crossdock_pdb"] = str(reference_path)
        snapshot["runtime"]["reference_id"] = reference_id

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "config_settings.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)


def output_dir_for_complex(output_root: Path, complex_id: str) -> Path:
    """Return the output directory for one complex."""
    return output_root / complex_id


def redock_output_root_for(input_root: Path) -> Path:
    """Return the canonical redock output directory for a dataset."""
    return input_root.parent / "prepared_inputs" / "redock"


def redock_receptor_path(redock_root: Path, complex_id: str) -> Path:
    """Return the redock-prepared receptor path for one complex."""
    return (
        redock_root / complex_id / RECEPTOR_PREPARED_NAME.format(complex_id=complex_id)
    )


def require_redock_outputs(input_root: Path) -> Path:
    """Require the canonical redock output directory."""
    redock_root = redock_output_root_for(input_root)
    if not redock_root.is_dir():
        raise FileNotFoundError(
            f"Redock output directory not found: {redock_root}\n"
            "Run redock preparation before cross-docking."
        )
    return redock_root


def require_redock_target_receptors(redock_root: Path, targets: list[Path]) -> None:
    """Require redock-prepared receptors for every selected target."""
    missing = [
        redock_receptor_path(redock_root, target.name)
        for target in targets
        if not redock_receptor_path(redock_root, target.name).is_file()
    ]
    if not missing:
        return

    preview = "\n".join(f"  {path}" for path in missing[:10])
    suffix = "\n  ..." if len(missing) > 10 else ""
    raise FileNotFoundError(
        f"Missing redock-prepared receptors for selected targets:\n{preview}{suffix}"
    )


def iter_complex_dirs(input_root: Path, complex_id: str | None = None) -> list[Path]:
    """Return one requested complex directory or all complex directories."""
    if complex_id is not None:
        path = input_root / complex_id
        if not path.is_dir():
            raise FileNotFoundError(f"Complex directory not found: {path}")
        return [path]

    return sorted(path for path in input_root.iterdir() if path.is_dir())


def chain_contact_stats(
    fixer: PDBFixer,
    ligand_in: Path,
    cutoff_nm: float,
) -> list[tuple[str, int, float]]:
    """Return chain ID, contact-residue count and minimum ligand distance."""
    mol = Chem.MolFromMolFile(str(ligand_in), removeHs=True)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Failed to read ligand: {ligand_in}")

    conf = mol.GetConformer()
    ligand_coords = (
        np.array(
            [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=float,
        )
        * 0.1
    )

    stats: list[tuple[str, int, float]] = []
    for chain in fixer.topology.chains():
        contact_residues: set[int] = set()
        min_dist = float("inf")

        for residue in chain.residues():
            atom_indices = [
                atom.index
                for atom in residue.atoms()
                if atom.element is not None and atom.element.atomic_number != 1
            ]
            if not atom_indices:
                continue

            protein_coords = np.array(
                [
                    fixer.positions[i].value_in_unit(unit.nanometer)
                    for i in atom_indices
                ],
                dtype=float,
            )
            distances = np.linalg.norm(
                ligand_coords[:, None, :] - protein_coords[None, :, :],
                axis=2,
            )
            residue_min = float(np.min(distances))
            min_dist = min(min_dist, residue_min)
            if residue_min <= cutoff_nm:
                contact_residues.add(residue.index)

        if min_dist < float("inf"):
            stats.append((chain.id, len(contact_residues), min_dist))

    return stats


def select_chain_ids(
    fixer: PDBFixer,
    ligand_in: Path,
    chain_policy: ChainPolicy,
    cutoff_angstrom: float,
    receptor_label: str,
) -> list[str]:
    """Select receptor chains according to the configured policy."""
    chain_ids = [chain.id for chain in fixer.topology.chains()]
    if chain_policy == "all" or len(chain_ids) <= 1:
        return chain_ids

    cutoff_nm = cutoff_angstrom * 0.1
    stats = chain_contact_stats(fixer, ligand_in, cutoff_nm)
    if not stats:
        raise ValueError(f"{receptor_label}: no protein chains with heavy atoms found")

    if chain_policy == "single":
        return [max(stats, key=lambda item: (item[1], -item[2], item[0]))[0]]

    if chain_policy == "nearby":
        keep = [chain_id for chain_id, _, min_dist in stats if min_dist <= cutoff_nm]
        if not keep:
            raise ValueError(
                f"{receptor_label}: no chains within {cutoff_angstrom:.2f} A of ligand"
            )
        return keep

    raise ValueError(f"Unknown chain policy: {chain_policy}")


def prepare_ligand(
    ligand_in: Path,
    protonate: bool,
    ph: float,
    output_path: Path | None = None,
) -> Path:
    """Optionally protonate a ligand and return the conformer-generation input."""
    if not ligand_in.is_file():
        raise FileNotFoundError(f"Ligand input not found: {ligand_in}")
    if not protonate:
        return ligand_in
    if output_path is None:
        raise ValueError("output_path is required when ligand protonation is enabled")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["obabel", str(ligand_in), "-O", str(output_path), "-p", str(ph)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"OpenBabel did not produce a valid ligand file: {output_path}"
        )
    return output_path


def prepare_protein(
    protein_in: Path,
    protein_out: Path,
    ph: float,
    forcefield_name: str,
    ligand_in: Path | None = None,
    chain_policy: ChainPolicy = "all",
    cutoff_angstrom: float = 6.0,
) -> None:
    """Prepare a receptor PDB with PDBFixer."""
    if not protein_in.is_file():
        raise FileNotFoundError(f"Protein input not found: {protein_in}")

    fixer = PDBFixer(filename=str(protein_in))
    if ligand_in is not None and chain_policy != "all":
        keep_chain_ids = select_chain_ids(
            fixer=fixer,
            ligand_in=ligand_in,
            chain_policy=chain_policy,
            cutoff_angstrom=cutoff_angstrom,
            receptor_label=protein_in.stem,
        )
        chain_ids = [chain.id for chain in fixer.topology.chains()]
        fixer.removeChains(
            chainIds=[cid for cid in chain_ids if cid not in keep_chain_ids]
        )

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=ph, forcefield=ForceField(forcefield_name))

    protein_out.parent.mkdir(parents=True, exist_ok=True)
    with protein_out.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)


def generate_conformers(
    ligand_in: Path,
    output_dir: Path,
    complex_id: str,
    num_confs: int,
    seed: int,
    minimise_conformers: MinimiseMode,
) -> None:
    """Generate 3D ligand conformers with RDKit."""
    if not ligand_in.is_file():
        raise FileNotFoundError(f"Ligand input not found: {ligand_in}")

    mol = Chem.MolFromMolFile(str(ligand_in), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit failed to read molecule from {ligand_in}")

    Chem.SanitizeMol(mol)
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    conf_ids = list(
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed)
    )
    if len(conf_ids) != num_confs:
        raise RuntimeError(
            f"RDKit generated {len(conf_ids)} of {num_confs} requested conformers for "
            f"{complex_id}"
        )

    if minimise_conformers == "uff":
        AllChem.UFFOptimizeMoleculeConfs(mol)
    elif minimise_conformers == "mmff94":
        properties = AllChem.MMFFGetMoleculeProperties(mol)
        if properties is None:
            raise ValueError(f"MMFF94 parameters unavailable for {ligand_in}")
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94")

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, conformer in enumerate(mol.GetConformers(), start=1):
        sdf_out = output_dir / LIGAND_CONFORMER_NAME.format(complex_id=complex_id, i=i)
        with Chem.SDWriter(str(sdf_out)) as writer:
            writer.write(mol, confId=conformer.GetId())


def process_complex(
    complex_path: Path,
    output_root: Path,
    num_confs: int,
    seed: int,
    chain_policy: ChainPolicy,
    ligand_cutoff_angstrom: float,
    protonate_ligand: bool,
    minimise_conformers: MinimiseMode,
    ph: float,
    forcefield_name: str,
) -> WorkerResult:
    """Prepare one complex for redocking."""
    complex_id = complex_path.name
    output_dir = output_dir_for_complex(output_root, complex_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ligand_in = complex_path / f"{complex_id}_ligand.sdf"
        protein_in = complex_path / f"{complex_id}_delig-desolv.pdb"
        if not ligand_in.is_file():
            raise FileNotFoundError(f"Missing ligand: {ligand_in}")
        if not protein_in.is_file():
            raise FileNotFoundError(f"Missing protein: {protein_in}")

        shutil.copy2(ligand_in, output_dir / f"{complex_id}_reference_ligand.sdf")
        ligand_for_conformers = prepare_ligand(
            ligand_in=ligand_in,
            protonate=protonate_ligand,
            ph=ph,
            output_path=output_dir / LIGAND_H_NAME.format(complex_id=complex_id),
        )
        prepare_protein(
            protein_in=protein_in,
            protein_out=output_dir
            / RECEPTOR_PREPARED_NAME.format(complex_id=complex_id),
            ph=ph,
            forcefield_name=forcefield_name,
            ligand_in=ligand_in,
            chain_policy=chain_policy,
            cutoff_angstrom=ligand_cutoff_angstrom,
        )
        generate_conformers(
            ligand_in=ligand_for_conformers,
            output_dir=output_dir,
            complex_id=complex_id,
            num_confs=num_confs,
            seed=seed,
            minimise_conformers=minimise_conformers,
        )
        return complex_id, True, None
    except Exception as exc:
        return complex_id, False, str(exc)


def align_pdb_to_reference(
    mobile_pdb: Path,
    reference_pdb: Path,
    output_pdb: Path,
) -> Path:
    """Align a mobile receptor onto a reference receptor using ProDy."""
    if not mobile_pdb.is_file():
        raise FileNotFoundError(f"Mobile PDB not found: {mobile_pdb}")
    if not reference_pdb.is_file():
        raise FileNotFoundError(f"Reference PDB not found: {reference_pdb}")

    mobile = matchAlign(parsePDB(str(mobile_pdb)), parsePDB(str(reference_pdb)))[0]
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    writePDB(str(output_pdb), mobile)
    return output_pdb


def process_crossdock_alignment(
    complex_dir: Path,
    redock_root: Path,
    output_root: Path,
    crossdock_prepared: Path,
    output_name: str,
) -> WorkerResult:
    """Align one prepared receptor into one redock-prepared complex frame."""
    complex_id = complex_dir.name
    try:
        align_pdb_to_reference(
            mobile_pdb=crossdock_prepared,
            reference_pdb=redock_receptor_path(redock_root, complex_id),
            output_pdb=output_root / complex_id / output_name,
        )
        return complex_id, True, None
    except Exception as exc:
        return complex_id, False, str(exc)


def run_parallel(
    items: list[Path],
    worker: Callable[[Path], WorkerResult],
    desc: str,
    num_workers: int,
) -> list[tuple[str, str]]:
    """Run a per-complex worker in parallel and collect structured failures."""
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, item) for item in items]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=desc, unit="complex"
        ):
            complex_id, ok, reason = future.result()
            if not ok:
                failures.append((complex_id, reason or "Unknown error"))
                print(f"[error] {complex_id}: {reason}")
    return failures


def write_failures(failures: list[tuple[str, str]], fail_file: Path) -> None:
    """Write per-complex failures to a TSV file."""
    if not failures:
        fail_file.unlink(missing_ok=True)
        return

    fail_file.parent.mkdir(parents=True, exist_ok=True)
    with fail_file.open("w", encoding="utf-8") as handle:
        handle.write("complex_id\treason\n")
        for complex_id, reason in failures:
            clean_reason = reason.replace("\n", " ").replace("\t", " ")
            handle.write(f"{complex_id}\t{clean_reason}\n")


def as_bool_series(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to a boolean Series."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return (
        series.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})
    )


def canonicalise_smiles(smiles: str) -> str | None:
    """Return canonical isomeric SMILES, or ``None`` if parsing fails."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def complex_prefix(complex_name: str) -> str:
    """Return the OpenBind duplicate-group prefix for a complex ID."""
    match = re.match(r"^(.*-x\d+)[a-z]$", complex_name)
    return match.group(1) if match is not None else complex_name


def load_annotation_table(input_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load and validate the standard OpenBind complex annotation table."""
    fragment_cfg = config["fragment_selection"]
    csv_path = Path(str(fragment_cfg["csv_path"])).expanduser()
    annotation_path = (
        csv_path if csv_path.is_absolute() else input_root.parent / csv_path
    )
    annotation_path = annotation_path.resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Fragment annotation CSV not found: {annotation_path}")

    df = pd.read_csv(annotation_path)
    required = {"complex_name", "smiles", "fragment_screen", "pb_valid", "artefact"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Annotation CSV is missing required columns: {sorted(missing)}"
        )
    if df["complex_name"].duplicated().any():
        duplicates = df.loc[df["complex_name"].duplicated(), "complex_name"].unique()
        raise ValueError(f"Duplicate complex_name entries: {duplicates[:10].tolist()}")

    df = df.copy()
    df["complex_name"] = df["complex_name"].astype(str)
    df["canonical_smiles"] = df["smiles"].map(canonicalise_smiles)
    invalid = df.loc[df["canonical_smiles"].isna(), "complex_name"].tolist()
    if invalid:
        raise ValueError(f"Invalid SMILES for complexes: {invalid[:10]}")
    return df


def load_fragment_candidates(
    input_root: Path,
    redock_root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load fragment candidates and require their prepared redock receptors."""
    fragment_cfg = config["fragment_selection"]
    annotation_df = load_annotation_table(input_root, config)
    mask = as_bool_series(annotation_df["fragment_screen"])

    if bool(fragment_cfg["pb_valid_filter"]):
        mask &= as_bool_series(annotation_df["pb_valid"])
    if bool(fragment_cfg["artefact_filter"]):
        mask &= ~as_bool_series(annotation_df["artefact"])

    candidate_df = annotation_df.loc[mask, ["complex_name", "canonical_smiles"]].copy()
    candidate_df = candidate_df.drop_duplicates(subset=["complex_name"]).reset_index(
        drop=True
    )
    if candidate_df.empty:
        raise ValueError("No fragment candidates remain after annotation filtering")

    missing_receptors = [
        redock_receptor_path(redock_root, str(complex_id))
        for complex_id in candidate_df["complex_name"]
        if not redock_receptor_path(redock_root, str(complex_id)).is_file()
    ]
    if missing_receptors:
        preview = "\n".join(f"  {path}" for path in missing_receptors[:10])
        suffix = "\n  ..." if len(missing_receptors) > 10 else ""
        raise FileNotFoundError(
            "Missing redock-prepared receptors for fragment candidates:\n"
            f"{preview}{suffix}"
        )

    print(f"[fragment_crossdock] total annotation rows: {len(annotation_df)}")
    print(
        f"[fragment_crossdock] fragment candidates after filtering: {len(candidate_df)}"
    )
    return candidate_df, annotation_df


def build_duplicate_exclusion_dict(
    annotation_df: pd.DataFrame,
    candidate_ids: set[str],
    duplicate_filter_mode: DuplicateMode,
) -> dict[str, set[str]]:
    """Build per-complex fragment exclusions by prefix and/or canonical SMILES."""
    complex_ids = annotation_df["complex_name"].astype(str).tolist()
    exclusions: dict[str, set[str]] = {complex_id: set() for complex_id in complex_ids}
    if duplicate_filter_mode == "none":
        return exclusions

    if duplicate_filter_mode in {"prefix", "both"}:
        prefix_groups: dict[str, set[str]] = defaultdict(set)
        for complex_id in complex_ids:
            prefix_groups[complex_prefix(complex_id)].add(complex_id)
        for complex_id in complex_ids:
            exclusions[complex_id].update(
                prefix_groups[complex_prefix(complex_id)] & candidate_ids
            )

    if duplicate_filter_mode in {"smiles", "both"}:
        smiles_groups: dict[str, set[str]] = defaultdict(set)
        smiles_by_complex: dict[str, str] = {}
        for row in annotation_df[["complex_name", "canonical_smiles"]].itertuples(
            index=False
        ):
            complex_id = str(row.complex_name)
            smiles = str(row.canonical_smiles)
            smiles_by_complex[complex_id] = smiles
            smiles_groups[smiles].add(complex_id)
        for complex_id, smiles in smiles_by_complex.items():
            exclusions[complex_id].update(smiles_groups[smiles] & candidate_ids)

    return exclusions


def process_fragment_alignment(
    task: tuple[Path, str, float],
    redock_root: Path,
    output_root: Path,
) -> tuple[str, str, float, bool, str]:
    """Align one selected fragment receptor into its target complex frame."""
    complex_path, fragment_id, similarity = task
    complex_id = complex_path.name

    output_name = f"fragment_crossdock_{fragment_id}_{similarity:.2f}_aligned.pdb"

    complex_output_dir = output_root / complex_id
    new_output_path = complex_output_dir / output_name

    result_complex_id, ok, reason = process_crossdock_alignment(
        complex_dir=complex_path,
        redock_root=redock_root,
        output_root=output_root,
        crossdock_prepared=redock_receptor_path(redock_root, fragment_id),
        output_name=output_name,
    )

    if ok:
        for old_path in complex_output_dir.glob("fragment_crossdock_*_aligned.pdb"):
            if old_path != new_output_path:
                old_path.unlink()

    return (
        result_complex_id,
        fragment_id,
        similarity,
        ok,
        reason or "",
    )


def run_fragment_crossdock(
    input_root: Path,
    output_root: Path,
    targets: list[Path],
    config: dict[str, Any],
    num_workers: int,
) -> list[tuple[str, str]]:
    """Select and align the most similar allowed fragment receptor per target."""
    fragment_cfg = config["fragment_selection"]
    duplicate_mode: DuplicateMode = fragment_cfg["duplicate_filter_mode"]
    redock_root = require_redock_outputs(input_root)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    candidate_df, annotation_df = load_fragment_candidates(
        input_root, redock_root, config
    )
    fragment_ids = candidate_df["complex_name"].astype(str).tolist()
    fragment_fps = [
        generator.GetFingerprint(Chem.MolFromSmiles(str(smiles)))
        for smiles in candidate_df["canonical_smiles"]
    ]
    exclusions = build_duplicate_exclusion_dict(
        annotation_df=annotation_df,
        candidate_ids=set(fragment_ids),
        duplicate_filter_mode=duplicate_mode,
    )
    smiles_by_complex = annotation_df.set_index("complex_name")[
        "canonical_smiles"
    ].to_dict()

    rows: list[list[Any]] = []
    failures: list[tuple[str, str]] = []
    alignment_tasks: list[tuple[Path, str, float]] = []

    for complex_path in tqdm(
        targets, desc="Selecting fragment receptors", unit="complex"
    ):
        complex_id = complex_path.name
        if complex_id not in smiles_by_complex:
            raise ValueError(
                f"Complex '{complex_id}' is missing from the annotation CSV"
            )

        target_mol = Chem.MolFromSmiles(str(smiles_by_complex[complex_id]))
        if target_mol is None:
            raise ValueError(f"Failed to parse annotation SMILES for {complex_id}")

        similarities = DataStructs.BulkTanimotoSimilarity(
            generator.GetFingerprint(target_mol), fragment_fps
        )
        allowed = [
            index
            for index, fragment_id in enumerate(fragment_ids)
            if fragment_id not in exclusions[complex_id]
        ]
        if not allowed:
            reason = "no fragment candidates remain after duplicate filtering"
            rows.append([complex_id, "", np.nan, False, reason])
            failures.append((complex_id, reason))
            print(f"[error] {complex_id}: {reason}")
            continue

        best_index = max(allowed, key=similarities.__getitem__)
        alignment_tasks.append(
            (complex_path, fragment_ids[best_index], float(similarities[best_index]))
        )

    worker = partial(
        process_fragment_alignment,
        redock_root=redock_root,
        output_root=output_root,
    )
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, task) for task in alignment_tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Aligning fragment receptors",
            unit="complex",
        ):
            complex_id, fragment_id, similarity, ok, reason = future.result()
            rows.append([complex_id, fragment_id, similarity, ok, reason])
            if not ok:
                failures.append((complex_id, reason or "Unknown error"))
                print(f"[error] {complex_id}: {reason}")

    output_root.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(
        rows,
        columns=[
            "complex_id",
            "most_similar_fragment",
            "ECFP4_Tanimoto_Similarity",
            "alignment_ok",
            "failure_reason",
        ],
    ).sort_values("complex_id")
    result_df.to_csv(output_root / "fragment_followon_similarity.csv", index=False)
    return failures


def prepare_complexes(
    targets: list[Path],
    output_root: Path,
    config: dict[str, Any],
    num_workers: int,
) -> list[tuple[str, str]]:
    """Prepare receptors and ligand conformers for redocking."""
    protein_cfg = config["protein"]
    ligand_cfg = config["ligand"]
    worker = partial(
        process_complex,
        output_root=output_root,
        num_confs=int(ligand_cfg["num_conformers"]),
        seed=int(ligand_cfg["seed"]),
        chain_policy=protein_cfg["chain_policy"],
        ligand_cutoff_angstrom=float(protein_cfg["ligand_cutoff_angstrom"]),
        protonate_ligand=bool(ligand_cfg["protonate"]),
        minimise_conformers=ligand_cfg["minimise_conformers"],
        ph=float(protein_cfg["pH"]),
        forcefield_name=str(protein_cfg["forcefield"]),
    )
    return run_parallel(targets, worker, "Preparing complexes", num_workers)


def print_runtime_summary(
    config: dict[str, Any],
    config_path: Path,
    input_root: Path,
    output_root: Path,
    target_count: int,
    reference_path: Path | None,
) -> None:
    """Print the resolved preparation settings."""
    mode = str(config["mode"])
    print("=" * 60)
    print("OpenBind input preparation")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Mode: {mode}")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Complexes selected: {target_count}")

    if mode == "redock":
        protein_cfg = config["protein"]
        ligand_cfg = config["ligand"]
        print(f"Chain policy: {protein_cfg['chain_policy']}")
        print(f"Protein pH: {protein_cfg['pH']}")
        print(f"Protein force field: {protein_cfg['forcefield']}")
        print(f"Ligand protonation: {ligand_cfg['protonate']}")
        print(f"Conformer minimisation: {ligand_cfg['minimise_conformers']}")
        print(f"Ligand conformers: {ligand_cfg['num_conformers']}")
        print(f"Ligand seed: {ligand_cfg['seed']}")
    elif mode == "crossdock":
        print(f"Protein pH: {config['protein']['pH']}")
        print(f"Protein force field: {config['protein']['forcefield']}")
        print(f"Cross-docking receptor: {reference_path}")
    else:
        fragment_cfg = config["fragment_selection"]
        print(f"Annotation file: {fragment_cfg['csv_path']}")
        print(f"PoseBusters filter: {fragment_cfg['pb_valid_filter']}")
        print(f"Artefact filter: {fragment_cfg['artefact_filter']}")
        print(f"Duplicate filter: {fragment_cfg['duplicate_filter_mode']}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Reference-data directory containing one subdirectory per complex.",
    )
    parser.add_argument(
        "--config",
        default="redock",
        help="Preparation YAML path or built-in profile name (default: redock).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the prepared-input output directory.",
    )
    parser.add_argument(
        "--complex-id",
        default=None,
        help="Prepare one complex instead of the full dataset.",
    )
    parser.add_argument(
        "--num-workers",
        "-j",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: available CPUs).",
    )
    parser.add_argument(
        "--crossdock-pdb",
        type=Path,
        default=None,
        help="Reference receptor PDB required by the crossdock profile.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help=(
            "Override fragment_selection.csv_path for fragment cross-docking. "
            "Useful when the annotation table is stored outside the dataset root."
        ),
    )
    return parser


def main() -> None:
    """Run input preparation."""
    args = build_parser().parse_args()
    confProDy(verbosity="none")

    input_root = args.input_dir.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    if args.annotation_file is not None:
        if config.get("mode") != "fragment_crossdock":
            raise ValueError(
                "--annotation-file is only valid with fragment_crossdock mode"
            )
        annotation_file = args.annotation_file.expanduser().resolve()
        if not annotation_file.is_file():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        config["fragment_selection"]["csv_path"] = str(annotation_file)
    validate_config(config)
    if (
        config["mode"] == "redock"
        and bool(config["ligand"]["protonate"])
        and shutil.which("obabel") is None
    ):
        raise FileNotFoundError("OpenBabel executable 'obabel' was not found on PATH")

    output_root, reference_path, reference_id = resolve_output_root(
        input_root=input_root,
        config=config,
        output_dir=args.output_dir,
        crossdock_pdb=args.crossdock_pdb,
    )
    targets = iter_complex_dirs(input_root, args.complex_id)
    if not targets:
        raise ValueError(f"No complex directories found in {input_root}")

    mode = str(config["mode"])
    redock_root: Path | None = None
    if mode in {"crossdock", "fragment_crossdock"}:
        redock_root = require_redock_outputs(input_root)
        require_redock_target_receptors(redock_root, targets)

    write_runtime_config(
        config=config,
        input_root=input_root,
        output_root=output_root,
        config_path=config_path,
        reference_path=reference_path,
        reference_id=reference_id,
    )
    print_runtime_summary(
        config=config,
        config_path=config_path,
        input_root=input_root,
        output_root=output_root,
        target_count=len(targets),
        reference_path=reference_path,
    )

    workers = 1 if args.complex_id else args.num_workers
    failures: list[tuple[str, str]]

    if mode == "redock":
        failures = prepare_complexes(targets, output_root, config, workers)
        failure_file = output_root / "failed_complexes.tsv"
    elif mode == "crossdock":
        assert (
            redock_root is not None
            and reference_path is not None
            and reference_id is not None
        )
        prepared_reference = output_root / f"{reference_id}_prepared.pdb"
        prepare_protein(
            protein_in=reference_path,
            protein_out=prepared_reference,
            ph=float(config["protein"]["pH"]),
            forcefield_name=str(config["protein"]["forcefield"]),
        )
        failures = run_parallel(
            items=targets,
            worker=partial(
                process_crossdock_alignment,
                redock_root=redock_root,
                output_root=output_root,
                crossdock_prepared=prepared_reference,
                output_name=f"{reference_id}_prepared_aligned.pdb",
            ),
            desc="Aligning cross-docking receptor",
            num_workers=workers,
        )
        failure_file = output_root / "failed_crossdock_alignments.tsv"
    else:
        failures = run_fragment_crossdock(
            input_root, output_root, targets, config, workers
        )
        failure_file = output_root / "failed_fragment_crossdock_alignments.tsv"

    write_failures(failures, failure_file)
    if failures:
        raise SystemExit(f"Preparation completed with {len(failures)} failed complexes")


if __name__ == "__main__":
    main()
