<img src="docs/images/carps_Logo_wide.png" alt="Logo"/>

# CARP-S
Welcome to CARP-S! 
This repository contains a benchmarking framework for optimizers.
It allows flexibly combining optimizers and benchmarks via a simple interface, and logging experiment results 
and trajectories to a database.
carps can launch experiment runs in parallel by using [hydra](https://hydra.cc), which offers launchers for slurm/submitit, Ray, RQ, and joblib.

The main topics of this README are:
- [Installation](#installation)
- [Minimal Example](#minimal-example)
- [Commands](#commands)
- [Adding a new Optimizer or Benchmark](#adding-a-new-optimizer-or-benchmark)

For more details on CARP-S, please have a look at the 
[documentation](https://AutoML.github.io/CARP-S/latest/) or our [blog post](https://automl.space/carps-a-framework-for-comparing-n-hyperparameter-optimizers-on-m-benchmarks/).

## Installation

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

If you want to install CARP-S for development, you can use the following command:
```bash
make install-dev
```
#### Apptainer
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


For example, if you want to use the `SMAC3` optimizer and the `BBOB` benchmark, you need to install the
requirements for both of them via:

```bash
# Install options for optimizers and benchmarks (these are Makefile commands, check the Makefile for more commands)
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
Check the `Makefile` in carps for more details.


## Minimal Example
Once the requirements for both an optimizer and a benchmark, e.g. `SMAC2.0` and `BBOB`, are installed, you can run
one of the following minimal examples to benchmark `SMAC2.0` on `BBOB` directly with Hydra:

```bash
# Run SMAC BlackBoxFacade on certain BBOB task
python -m carps.run +optimizer/smac20=blackbox +task/BBOB=cfg_4_1_0 seed=1 task.optimization_resources.n_trials=25

# Run SMAC BlackBoxFacade on all available BBOB tasks for 10 seeds
python -m carps.run +optimizer/smac20=blackbox '+task/BBOB=glob(*)' 'seed=range(1,11)' -m
```

For the second command, the Hydra -m (or --multirun) option indicates that multiple runs will be 
performed over a range of parameter values. In this case, it's indicating that the benchmarking
should be run for all available BBOB tasks (`+task/BBOB=glob(*)`) and for 10 different 
seed values (seed=range(1,11)).

## Commands

You can run a certain task and optimizer combination directly with Hydra via:
```bash
python -m carps.run +task=... +optimizer=... seed=... -m
```

To check whether any runs are missing, you can use the following command. It will create
a file `runcommands_missing.sh` containing the missing runs:
```bash
python -m carps.utils.check_missing <rundir>
```

To collect all run data generated by the file logger into csv files, use the following command:
```bash
python -m carps.analysis.gather_data <rundir>
```
The csv files are then located in `<rundir>`. `logs.csv` contain the trial info and values and 
`logs_cfg.csv` contain the experiment  configuration.
The experiments can be matched via the column `experiment_id`.

## CARPS and MySQL Database
Per default, `carps` logs to files. This has its caveats: Checking experiment status is a bit more cumbersome (but 
possible with `python -m carps.utils.check_missing <rundir>` to check for missing/failed experiments) and reading from
the filesystem takes a long time. For this reason, we can also control and log experiments to a MySQL database with
`PyExperimenter`.

### Requirements and Configuration
Requirement: MySQL database is set up.

1. Add a `credentials.yaml` file in `carps/experimenter` with the following content:
```yaml
CREDENTIALS:
  Database:
      user: someuser
      password: amazing_password
  Connection:
      Standard:
        server: mysql_server
        port: 3306 (most likely)
```
2. Edit `carps/experimenter/py_experimenter.yaml` by setting:
```yaml
PY_EXPERIMENTER:
  n_jobs: 1

  Database:
    use_ssh_tunnel: false
    provider: mysql
    database: your_database_name
...
```
!!! Note: If you use an ssh tunnel, set `use_ssh_tunnel` to `true` in `carps/experimenter/py_experimenter.yaml`.
Set up  `carps/experimenter/credentials.yaml` like this:
```yaml
CREDENTIALS:
  Database:
      user: someuser
      password: amazing_password
  Connection:
      Standard:
        server: mysql_server
        port: 3306 (most likely)
      Ssh:
        server: 127.0.0.1
        address: some_host  # hostname as specified in ~/.ssh/config
        # ssh_private_key_password: null
        # server: example.sshmysqlserver.com (address from ssh server)
        # address: example.sslserver.com
        # port: optional_ssh_port
        # remote_address: optional_mysql_server_address
        # remote_port: optional_mysql_server_port
        # local_address: optional_local_address
        # local_port: optional_local_port
        # passphrase: optional_ssh_passphrase
```
### Create Experiments
First, in order for PyExperimenter to be able to pull experiments from the database, we need to fill it.
The general command looks like this:
```bash
python -m carps.experimenter.create_cluster_configs +task=... +optimizer=... -m
```
All subset runs were created with `scripts/create_experiments_in_db.sh`.

### Running Experiments
Now, execute experiments with:
```bash
python -m carps.run_from_db 'job_nr_dummy=range(1,1000)' -m
```
This will create 1000 multirun jobs, each pulling an experiment from PyExperimenter and executing it.

!!! Note: On most slurm clusters the max array size is 1000.
!!! Note: On our mysql server location, at most 300 connections at the same time are possible. You can limit your number
    of parallel jobs with `hydra.launcher.array_parallelism=250`.
!!! `carps/configs/runfromdb.yaml` configures the run and its resources. Currently defaults for our slurm cluster are
    configured. If you run on a different cluster, adapt `hydra.launcher`.

Experiments with error status (or any other status) can be reset via:
```bash
python -m carps.experimenter.database.reset_experiments
```

### Get the results from the database and post-process


## Adding a new Optimizer or Benchmark
For instructions on how to add a new optimizer or benchmark, please refer to the contributing 
guidelines for 
[benchmarks](https://automl.github.io/CARP-S/latest/contributing/contributing-a-benchmark/)
and
[optimizers](https://automl.github.io/CARP-S/latest/contributing/contributing-an-optimizer/).

## Using your (external) optimizer or benchmark
In the case when you are developing your optimizer or benchmark in a standalone package, you can use carps without directly working in the carps repo.
For a custom benchmark we have an [example repo](https://github.com/automl/OptBench).
It shows how to use your own benchmark with carps optimizers.
For a custom optimizer check this [example repo](https://github.com/automl/CARP-S-example-optimizer).
Information is also available [here](https://automl.github.io/CARP-S/guides/using-carps/).

## Evaluation Results
For each task_type (blackbox, multi-fidelity, multi-objective and multi-fidelity-multi-objective) and set (dev and test), we run selected optimizers and provide the data.
Here we provide the link to the [meta data](https://drive.google.com/file/d/17pn48ragmWsyRC39sInsh2fEPUHP3BRT/view?usp=sharing) 
that contains the detailed optimization setting for each run  
and the [running results](https://drive.google.com/file/d/1yzJRbwRvdLbpZ9SdQN2Vk3yQSdDP_vck/view?usp=drive_link) that 
records the running results of each optimization-benchmark combination. 
