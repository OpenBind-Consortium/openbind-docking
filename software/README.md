# Software setup

This directory contains installation instructions for the third-party docking software used by the OpenBind docking workflow.

GNINA, Smina, and DiffDock are **not included** in this repository and must be installed separately. The paths to the installed software, optional Singularity/Apptainer images, and Slurm profiles are configured in [`../configs/docking/software_settings.yaml`](../configs/docking/software_settings.yaml).

## Expected directory structure

A local installation can follow this structure:

```text
software/
├── gnina/
│   ├── gnina
│   └── gnina_singularity.sif          # optional
├── smina/
│   └── smina
└── DiffDock/
    ├── inference.py
    ├── default_inference_args.yaml
    └── diffdock_singularity.sif        # optional
```

The exact locations are configurable, so this structure is recommended rather than required.

## Software and Slurm configuration

Update [`../configs/docking/software_settings.yaml`](../configs/docking/software_settings.yaml) after installation. For example:

```yaml
methods:
  diffdock:
    executable: software/DiffDock/
    singularity: software/DiffDock/diffdock_singularity.sif
    profile: gpu

  gnina:
    executable: software/gnina/gnina
    singularity: software/gnina/gnina_singularity.sif
    profile: gpu

  smina:
    executable: software/smina/smina
    profile: cpu
```

Relative paths are resolved from the OpenBind docking repository root.

The `profiles` section accepts Slurm options as key-value pairs and passes them through as `#SBATCH` directives. Add the settings required by your cluster, for example `partition`, `account`, `cluster`, or `qos`. Boolean `true` values are emitted as flag-only options, while `false`, empty, and `null` values are omitted.

```yaml
profiles:
  gpu:
    cpus-per-task: 8
    mem: 16G
    time: "24:00:00"
    gres: gpu:1
    account: my-project
    partition: gpu
    exclusive: true
```

No site-specific partition, account, or cluster is set in the repository defaults.

## Optional Singularity / Apptainer environments

Optional Singularity/Apptainer images can be used for GNINA and DiffDock. These images provide reproducible runtime dependency environments for the locally installed software; the executable or DiffDock source tree is still provided separately through `software_settings.yaml`.

They are primarily useful for:

- HPC environments;
- CUDA compatibility;
- avoiding dependency conflicts; and
- improving reproducibility across systems.

If a Singularity image is configured for a GPU Slurm profile, the OpenBind launcher adds GPU support when the profile requests a GPU resource.

## GNINA

**Recommended version:** `v1.3.2`

Create the software directory:

```bash
mkdir -p software/gnina
```

Download the precompiled GNINA binary:

```bash
wget https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2
mv gnina.1.3.2 software/gnina/gnina
chmod +x software/gnina/gnina
```

### Optional Singularity image

```bash
singularity pull software/gnina/gnina_singularity.sif \
    oras://ghcr.io/jnelen/gnina_singularity:v1
```

## Smina

Download Smina from its project page and place the executable at:

```text
software/smina/smina
```

Then make it executable:

```bash
chmod +x software/smina/smina
```

## DiffDock

Clone DiffDock into `software/`:

```bash
cd software
git clone https://github.com/gcorso/DiffDock.git
cd ..
```

Follow the official DiffDock installation instructions to install its Python dependencies and model files. For reproducible benchmark runs, record and reuse the exact DiffDock version or commit used for the calculation.

### Important: configure the number of generated poses

DiffDock reads inference settings from:

```text
software/DiffDock/default_inference_args.yaml
```

The value of `samples_per_complex` in this file **must match** `methods.diffdock.num_modes` in the OpenBind docking experiment YAML.

For the supplied A71EV2A benchmark configuration:

```yaml
methods:
  diffdock:
    num_modes: 40
```

set the DiffDock configuration to:

```yaml
samples_per_complex: 40
```

This manual change is required because the current DiffDock inference implementation loads `default_inference_args.yaml` after parsing command-line arguments. As a result, the YAML value for `samples_per_complex` overrides a value supplied on the command line.

The OpenBind runner checks this setting before starting DiffDock and raises a clear error if the two values do not match. This prevents a run configured for, for example, 40 poses from silently producing the DiffDock YAML default instead.

If you change `methods.diffdock.num_modes` for another experiment, update `samples_per_complex` in `default_inference_args.yaml` to the same value before submitting the jobs.

### Optional Singularity image

```bash
singularity pull software/DiffDock/diffdock_singularity.sif \
    oras://ghcr.io/jnelen/diffdock_singularity:v1
```

## Final checks

Before submitting a benchmark, confirm that:

- the configured executable and installation paths exist;
- any configured Singularity/Apptainer images exist;
- the required Slurm profile contains the settings needed by your cluster;
- GNINA and Smina binaries are executable; and
- DiffDock `samples_per_complex` matches `methods.diffdock.num_modes` in the experiment configuration.

For method-specific installation or advanced configuration, refer to the corresponding upstream documentation.
