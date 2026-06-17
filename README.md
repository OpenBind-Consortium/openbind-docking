# OpenBind-Docking

This repository contains the docking workflow used for the OpenBind structural benchmarks. It supports input preparation, docking, pose evaluation, and generation of standardised docking result tables for downstream benchmarking and analysis.

The workflow is designed to be reusable across OpenBind datasets. The **enteroviral 2A protease (A71EV2A)** dataset is used throughout this README as a concrete example and to demonstrate reproduction of the docking benchmark reported for the first OpenBind data release.

The default workflow uses **GNINA**, **Smina**, and **DiffDock** and is designed for execution on a Slurm cluster.

## Installation

First, clone the repository and enter the repository directory:

```bash
git clone https://github.com/OpenBind-Consortium/openbind-docking.git
cd openbind-docking
```

Then create and activate the OpenBind docking environment:

```bash
conda env create -f environment.yml
conda activate openbind-docking
```

GNINA, Smina, and DiffDock must be installed separately. Installation guidance is provided in [`software/README.md`](software/README.md). Their executable or installation paths, optional Singularity/Apptainer images, and Slurm profiles are configured in [`configs/docking/software_settings.yaml`](configs/docking/software_settings.yaml).

Before running the workflow, update these settings for your system. Slurm profile entries are passed through as `#SBATCH` directives, so cluster-specific options such as `partition`, `account`, `cluster`, or `qos` can be added as required. Empty, `false`, and `null` settings are ignored.

Docking experiments are defined separately from software settings. The exact settings used for the enteroviral 2A protease benchmark are provided in [`configs/docking/A71EV2A_benchmark.yaml`](configs/docking/A71EV2A_benchmark.yaml).

**DiffDock requires one additional installation-time setting:** `samples_per_complex` in `software/DiffDock/default_inference_args.yaml` must be at least as large as `methods.diffdock.num_modes` in the docking experiment configuration. The runner validates this before starting DiffDock. See [`software/README.md`](software/README.md) for details.

## Command-line conventions

The scripts use positional arguments for their primary input directory or file and named options for configuration and runtime overrides. Across the workflow:

* `--config` selects the relevant experiment or preparation YAML.
* `--software-settings` selects executable, container, and Slurm settings.
* `--output-dir` overrides the default output location.
* `--complex-id` selects a single complex where supported.
* `--num-workers` controls local multiprocessing where supported.

Scientific preparation settings such as protonation, chain selection, conformer generation, docking parameters, and numbers of retained benchmark poses are stored in YAML rather than duplicated as command-line overrides.

---

## Example dataset: enteroviral 2A protease (A71EV2A)

The enteroviral 2A protease dataset is used here as an example of how to run the complete workflow.

The prepared input dataset required to reproduce the docking benchmark from scratch is available from [Zenodo](https://zenodo.org/records/20798527). Download `A71EV2A_data.zip` directly with:

```bash
wget -O A71EV2A_data.zip "https://zenodo.org/records/20798528/files/A71EV2A_data.zip?download=1"
```

Extract it into the repository `datasets/` directory:

```bash
mkdir -p datasets
unzip A71EV2A_data.zip -d datasets/
```

The relevant structure is:

```text
datasets/A71EV2A_data/
├── data/
├── crossdock_proteins/
│   ├── A71EV2A_8POA.pdb
│   └── A71EV2A_AF.pdb
├── annotated_complexes.csv
└── ...
```

The `data/` directory contains the experimental protein-ligand complexes used to prepare the docking inputs. Each complex directory is expected to contain the corresponding crystallographic ligand and ligand- and solvent-removed receptor files:

```text
<complex_id>_ligand.sdf
<complex_id>_delig-desolv.pdb
```

Here, `<complex_id>` is the identifier used throughout the workflow, for example `A71EV2A-x5261a`.

Prepared inputs and docked poses are also available separately from [Zenodo](https://zenodo.org/records/20798527) as `A71EV2A_prepared_inputs.zip` and `A71EV2A_docked.zip` respectively, for users who want to inspect or reanalyse the docking calculations without rerunning the docking stage.

The final standardised pose-level output, `final_docking_pose_data.parquet`, is provided directly in the [enteroviral 2A protease benchmark repository](https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark). This is the file used by the downstream plotting and benchmarking scripts and is the most convenient starting point for reproducing the published figures.

---

# Reproducing the enteroviral 2A protease docking benchmark

The following commands demonstrate the complete workflow using `A71EV2A_data.zip`. All commands assume that the archive has been extracted to:

```text
datasets/A71EV2A_data/
```

For another dataset, the same workflow can be followed using the corresponding input data, cross-docking structures, annotations, and docking configuration.

If you do not want to rerun the input-preparation steps, the prepared inputs used for the benchmark can be downloaded directly from [Zenodo](https://zenodo.org/records/20798527) as `A71EV2A_prepared_inputs.zip`. Otherwise, follow the preparation steps below to generate them from the input dataset.

## 1. Prepare redocking inputs

Redocking must be prepared first because the resulting receptors and ligand inputs are also used by the cross-docking workflows.

```bash
python scripts/prepare_inputs.py \
    datasets/A71EV2A_data/data/ \
    --config redock \
    --num-workers 8
```

## 2. Prepare fragment-informed cross-docking

Fragment-informed cross-docking selects an experimentally determined fragment-bound receptor for each target compound.

```bash
python scripts/prepare_inputs.py \
    datasets/A71EV2A_data/data/ \
    --config fragment_crossdock \
    --num-workers 8
```

The fragment annotations are read from the path configured as `fragment_selection.csv_path` in `fragment_crossdock.yaml`. For the example dataset this resolves to `annotated_complexes.csv`. If the annotation table is stored elsewhere, override it at runtime with `--annotation-file path/to/annotated_complexes.csv`.

## 3. Prepare cross-docking against the 8POA apo receptor

```bash
python scripts/prepare_inputs.py \
    datasets/A71EV2A_data/data/ \
    --config crossdock \
    --crossdock-pdb datasets/A71EV2A_data/crossdock_proteins/A71EV2A_8POA.pdb \
    --num-workers 8
```

## 4. Prepare cross-docking against the AlphaFold2 apo receptor

```bash
python scripts/prepare_inputs.py \
    datasets/A71EV2A_data/data/ \
    --config crossdock \
    --crossdock-pdb datasets/A71EV2A_data/crossdock_proteins/A71EV2A_AF.pdb \
    --num-workers 8
```

Prepared inputs are written to:

```text
datasets/A71EV2A_data/prepared_inputs/
├── redock/
├── fragment_crossdock/
├── A71EV2A_8POA/
└── A71EV2A_AF/
```

Each preparation directory contains a `config_settings.yaml` snapshot recording the configuration and resolved runtime paths used for that run.

---

# Running docking

If you do not want to rerun the docking calculations, the docked poses used for the benchmark can be downloaded directly from [Zenodo](https://zenodo.org/records/20798527) as `A71EV2A_docked.zip`. These outputs can be used as the starting point for the analysis steps below. Otherwise, configure and run the docking calculations as described here.

Before submitting docking jobs, make sure that:

1. the required docking software is installed;
2. executable and container paths in [`configs/docking/software_settings.yaml`](configs/docking/software_settings.yaml) are correct;
3. the Slurm profiles in the same file match your cluster; and
4. the docking experiment configuration, such as [`configs/docking/A71EV2A_benchmark.yaml`](configs/docking/A71EV2A_benchmark.yaml), contains the settings you want to use for the run.

In particular, review the configured docking methods, numbers of generated poses, and analysis settings before launching the jobs. If using DiffDock, ensure that `samples_per_complex` in `software/DiffDock/default_inference_args.yaml` is at least as large as the configured `methods.diffdock.num_modes`.

Slurm profiles are passed through generically. Normal entries are rendered as `#SBATCH --key=value`, while boolean `true` is rendered as a flag-only `#SBATCH --key`; empty and `false` values are omitted. The supplied profiles therefore contain only portable baseline settings. Cluster-specific options such as `partition`, `account`, `cluster`, `qos`, or other supported Slurm settings can be added directly when required.

All docking calculations used for the enteroviral 2A protease benchmark can then be submitted using the supplied benchmark configuration:

```bash
python scripts/submit_docking_jobs.py \
    datasets/A71EV2A_data/prepared_inputs/ \
    --config configs/docking/A71EV2A_benchmark.yaml
```

Use `--dry-run` to print the generated Slurm scripts without submitting them.

Docking outputs are written to:

```text
datasets/A71EV2A_data/docking/
```

Docking jobs are safe to rerun after interruption. Existing non-empty GNINA/Smina outputs are skipped per ligand input, and DiffDock runs are skipped once the requested ranked poses are present. Existing outputs are only removed when `--overwrite` is explicitly requested.

The docking jobs must finish before starting the analysis stage.

---

# Analysing docking results

Once all docking calculations have completed, submit one analysis job for each configured method/mode combination:

```bash
python scripts/submit_analysis_jobs.py \
    datasets/A71EV2A_data/docking/ \
    --config configs/docking/A71EV2A_benchmark.yaml
```

The analysis evaluates generated poses using PoseBusters and OST-based structural metrics. The CPU Slurm profile in `software_settings.yaml` is used for these jobs, including the number of worker processes passed to `analyse_docking.py`.

Processed results are written to:

```text
datasets/A71EV2A_data/analysis/
├── csv/
├── parquet/
└── logs/
```

---

# Preparing the final docking result table

After all analysis jobs have completed, combine the individual results into the final standardised pose-level table:

```bash
python scripts/prepare_docking_results.py \
    datasets/A71EV2A_data/analysis/ \
    --config configs/docking/A71EV2A_benchmark.yaml
```

By default, the script reads `datasets/A71EV2A_data/annotated_complexes.csv`. An alternative annotation table can be supplied with `--annotation-file`.

The number of top poses retained for the final benchmark table is configured with `analysis.num_top_poses` in [`configs/docking/A71EV2A_benchmark.yaml`](configs/docking/A71EV2A_benchmark.yaml), rather than being hard-coded in the processing script.

This produces:

```text
datasets/A71EV2A_data/analysis/final_docking_pose_data.csv
datasets/A71EV2A_data/analysis/final_docking_pose_data.parquet
```

`final_docking_pose_data.parquet` is the final processed output of the docking workflow. For enteroviral 2A protease, this file can be used directly by the plotting and benchmarking scripts in the corresponding dataset repository.

