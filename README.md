# ⏱️ SMACBenchmarking 📐
Welcome to SMACBenchmarking! 
This repository contains a benchmarking framework for optimizers.
It allows flexibly combining optimizers and benchmarks via a simple interface, and logging experiment results 
and trajectories to a database.
There are two main ways to use this framework:
1. Run everything in the same environment (e.g. to test locally)
2. Build separate Singularity/ Apptainer containers for the optimizer and the benchmark (e.g. to run on a cluster)

The first options can allow for faster development, but the second option is more robust and flexible since python or 
other package versions don't clash, and eases execution on e.g. a SLURM cluster.

Main Topics of this README:
- [Conceptual Overview 🗺](#conceptual-overview-🗺)
- [Usage - Local Setup 📍](#usage---local-setup-📍)
- [Usage - Cluster Setup 🤖](#usage---cluster-setup-🤖)
- [Adding a new Optimizer or Benchmark 🆕](#adding-a-new-optimizer-or-benchmark-🆕)

## Conceptual Overview 🗺 
TODO Overview picture and description

## Usage - Local Setup 📍

### Installation
Create a conda environment and install the package.

```bash
git clone https://github.com/AutoML/SMACBenchmarking.git
cd SMACBenchmarking
conda create -n smacbenchmarking python=3.11
conda activate smacbenchmarking

# Install for usage
pip install .

# Install for development
make install-dev

pip install -r requirements.txt
```

Additionally, you need to install the requirements for the benchmark and optimizer that you want to use.
For example, if you want to use the `SMAC2.0` optimizer and the `BBOB` benchmark, you need to install the
requirements for both of them.

```bash
pip install -e ".[smac20]"
pip install -e ".[bbob]"
```

### Minimal Example
Once the requirements for both an optimizer and a benchmark, e.g. `SMAC2.0` and `BBOB`, are installed, you can run
one of the following minimal examples to benchmark `SMAC2.0` on `BBOB` directly with Hydra:

```bash
# Run SMAC BlackBoxFacade on certain BBOB problem
python smacbenchmarking/run.py +optimizer/smac20=blackbox +problem/BBOB=cfg_4_1_4_0 seed=1 task.n_trials=25

# Run SMAC BlackBoxFacade on all available BBOB problems for 10 seeds
python smacbenchmarking/run.py +optimizer/smac20=blackbox '+problem/BBOB=glob(*)' 'seed=range(1,11)'
```

Note that in this case, no logging is done.

## Usage - Cluster Setup 🤖
### Installation
Local: Install Apptainer

Cluster: Configure Singularity/ Apptainer

Setup Database if you want to log to database (mysql)

### Database
🚧 UNDER CONSTRUCTION 🚧

Current Status: SQLite is used as a database for testing. Once the database server is up and running, 
we will switch to MySQL.

Before you can start any jobs, the jobs need to be dispatched to the database.
To this end, call the file `create_cluster_configs.py` with the desired hydra arguments.
This can be done locally or on the server if you can execute python there directly.
If you execute it locally, the database file `smacbenchmarking.db` will be created in the current directory and 
needs to be transferred to the cluster.

```bash
python smacbenchmarking/container/create_cluster_configs.py +optimizer/DUMMY=config +problem/DUMMY=config 'seed=range(1,21)' --multirun
```

---
Eventually:

All results will be written to a central database.
This database needs to be set up once on the server.
MySQL can be installed with the information [here](https://dev.mysql.com/doc/refman/8.0/en/linux-installation.html).


Documentation at https://AutoML.github.io/SMACBenchmarking/main


## Containerization
To run benchmarking with containers, both the optimizer and benchmark have to be wrapped separately. 
We use Singularity/ Apptainer for this purpose.
The following example illustrates the principle based on a `DummyOptimizer` and `DummyBenchmark`.

#### Noctua2 Setup Before Compilation

Include the following lines in your `~/.bashrc`:

```bash
export SINGULARITY_CACHEDIR=$PC2PFS/hpc-prf-intexml/<USER>/.singularity_cache
export SINGULARITY_TMPDIR=/dev/shm/intexml<X>
mkdir /dev/shm/intexml<X> -p
```

### Optimizer
A Singularity recipe has to be created for the optimizer, which should be saved in the folder `container_recipes`.
This recipe has the purpose of setting up a container in which the optimizer can be run, e.g., installing the 
required packages, setting environment variables, copying files and so on.
For the `Dummy_Optimizer` this is `container_recipes/dummy_optimizer/dummy_optimizer.recipe`, which you can consult 
as a basis for other optimizers.

The optimizer then has to be built to an image named after the optimizer id, e.g., `DUMMY_Optimizer.sif` for the
`DummyOptimizer` using the following command:

```bash
singularity build DUMMY_Optimizer.sif container_recipes/DUMMY_Optimizer/DUMMY_Optimizer.recipe
```

To facilitate this process, a short script is provided for this purpose, which is however system-specific to Noctua2.
It can be run as follows:

```bash
./compile_noctua2.sh DUMMY_Optimizer.sif container_recipes/DUMMY_Optimizer/DUMMY_Optimizer.recipe
```

### Benchmark
Like for the optimizer, a Singularity recipe has to be created for the benchmark, which should be saved in the folder
`container_recipes` as well.

The benchmark image also has to be according to the benchmark id, e.g., `DUMMY_Problem.sif` for the 
`DummyBenchmark` 
using
the following command:

```bash
singularity build DUMMY_Problem.sif container_recipes/DUMMY_Problem/DUMMY_Problem.recipe
```

Command for Noctua2:

```bash
./compile_noctua2.sh DUMMY_Problem.sif container_recipes/DUMMY_Problem/DUMMY_Problem.recipe
```

### Running
A third container is needed that handles the hydra config. It does not need to be adjusted for each optimizer or
benchmark, but can be used as is. It can be built as follows:

```bash
singularity build runner.sif container_recipes/runner.recipe
```

Command for Noctua2:

```bash
./compile_noctua2.sh runner.sif container_recipes/runner.recipe
```

Running the containerized benchmarking system is also system-dependent. An example for Noctua2 is provided in the
script `start_container_noctua2.sh`. It can be run as follows:

```bash
./start_container_noctua2.sh
```

**NOTE**: This needs to be run in a SLURM-job, so either an interactive job

```bash
srun --cpus-per-task=2 -p normal --mem=2gb -n 1 --time=00:30:00 --pty bash
```

or a job allocated via script.

This will pull a job from the database and run it (database needs to be initialized beforehand).
To be efficient, this command should eventually be integrated into a SLURM script, which can be submitted to the
cluster (e.g. with job arrays).

### Example SLURM script
# TODO

### Overview of the whole process

![Overview of the whole process](images/smac_benchmarking_containers.drawio.png)


The overall benchmarking system works as follows: 

We have three different containers wrapping different functionality and a shell script controlling these containers. 
The `Runner (HydraInitializer)` container is responsible for pulling a PyExperimenter experiment from the database and 
writing files to the disk which are required to initialize the `Optimizer` and the `Benchmark` container. 
The `Benchmark` container wraps the actual benchmark to be run and provides two main functionalities via a web service. 
First, it allows to get the search space associated with the benchmark. 
Second, it answers requests providing a configuration to be evaluated with the corresponding evaluation result.
The `Optimizer` container wraps the optimizer to be benchmarked and interacts with the `Benchmark` container.
Any information required to boot the containers is written to the hard drive by the `HydraInitializer` container. 

Note that we provide wrappers for the optimizer and the benchmark interfaces such that when you implement an 
optimizer or a benchmark within our benchmarking framework, 
you can ignore all aspects of the system just described and simply follow the simple API. 

## Adding a new Optimizer or Benchmark 🆕
To add a new optimizer or benchmark to the repository you need to
1. Implement the optimizer or benchmark according to the corresponding interface
    - **Optimizer**
       - [Optimizer Interface](smacbenchmarking/optimizers/optimizer.py) <br> 
          put implementation in [optimizers](smacbenchmarking/optimizers)
       - [Benchmark Interface](smacbenchmarking/benchmarks/problem.py); put implementation in folder [benchmarks]
         (smacbenchmarking/benchmarks)
2. Add requirements for the optimizer or benchmark to the [setup.py](setup.py) under `extras-require`. 
   Please specify exact versions of all requirements! This is very important for reproducibility.
3. Add the config TODO
4. Add a howto TODO
