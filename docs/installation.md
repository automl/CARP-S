# Installation

### Installation from PyPI

To install CARP-S, you can simply use `pip`:

1. Create virtual env with conda or uv

```bash
# Conda
conda create -n carps python=3.12
conda activate carps

# -OR -

# uv
pip install uv
export PIP="uv pip"  # Env var needed for Makefile commands
uv venv --python=3.12 carpsenv
source carpsenv/bin/activate
```

2. Install  carps.
```bash
pip install carps
```
### Installation from Source

If you want to install from source, you can clone the repository and install CARP-S via:

#### Conda
```bash
git clone https://github.com/AutoML/CARP-S.git
cd CARP-S
export PIP="pip"
conda create -n carps python=3.12
conda activate carps

# Install for usage
$PIP install .
```

#### uv
```bash
git clone https://github.com/AutoML/CARP-S.git
cd CARP-S
pip install uv
export PIP="uv pip"
uv venv --python=3.12 carpsenv
source carpsenv/bin/activate

# Install for usage
$PIP install .

# Install as editable
$PIP install -e .
```

If you want to install CARP-S for development, you can use the following command (from the root of the repo):
```bash
$PIP install -e .
python -m carps.build.make install-dev
```
#### Apptainer
⚠ This is still experimental.

You can also create a container with the env setup by running `apptainer build container/env.sif container/env.def`.
Then you can execute any carps commands as usual by add this prefix `apptainer exec container/env.sif` before the
command, e.g. `apptainer exec container/env.sif python -m carps.run +task/... +optimizer/...`.
There is also an sbatch script to run experiments from the database using the apptainer on a slurm cluster
(`sbatch scripts/container_run_from_db.sh`). You might need to adapt the array size and the number of repetitions
according to the number of experiments you can run.

PS.: On some clusters you might need to load the module apptainer like so `module load tools Apptainer`.
Troubleshooting: If you have problems writing your cache directory, mount-bind it like so
`apptainer shell --bind $XDG_CACHE_HOME container/env.sif`. This binds the directory `$XDG_CACHE_HOME` in the
container to the directory `$XDG_CACHE_HOME` on the host.
If you have problems with `/var/lib/hpobench`, this bind might help: 
`<hpobench data dir>:/var/lib/hpobench/data`. `<hpobench data dir>` can be found in
[`.hpobenchrc`](https://github.com/automl/HPOBench/?tab=readme-ov-file#configure-hpobench).

#### A note on python versions
For python3.12, numpy should be `numpy>=2.0.0`. For python3.10, numpy must be `numpy==1.26.4`, you can simply
`pip install numpy==1.26.4` after running the proposed install commands.

### Installing Benchmarks and Optimizers

Additionally, you need to install the requirements for the benchmark and optimizer that you want to use.

⚠ You can specify the directory of the task data by `export CARPS_TASK_DATA_DIR=...`. Please use absolute dirnames.
The default location is `<carps package location>/task_data`. If you specify a custom dir, always export the env var.
(The carps package location is the root of the package, not of the repo.)

For example, if you want to use the `SMAC3` optimizer and the `BBOB` benchmark, you need to install the
requirements for both of them via:

```bash
# Install options for optimizers and benchmarks (these are Makefile commands, check the Makefile at carps/build for
# more commands)
# The commands should be separated by a whitespace
python -m carps.build.make benchmark_bbob optimizer_smac
```
The benchmarks and optimizers can all be installed in one environment (tested with python3.12).

All possible install options for benchmarks are:
```
benchmark_bbob benchmark_hpobench benchmark_hpob benchmark_mfpbench benchmark_pymoo benchmark_yahpo
```
⚠ Some benchmarks require to download surrogate models and/or containers and thus might take disk space and time to
download.

All possible install options for optimizers are:
```
optimizer_smac optimizer_dehb optimizer_nevergrad optimizer_optuna optimizer_ax optimizer_skopt optimizer_synetune
```
All of the above except `optimizer_hebo` work with python3.12.

You can also install all benchmarks in one go with `benchmarks` and all optimizers with `optimizers`.
Check the `carps/build/Makefile` in carps for more details.