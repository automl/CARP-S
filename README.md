# SMACBenchmarking



## Installation
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

Documentation at https://AutoML.github.io/SMACBenchmarking/main

## Minimal Example

```bash
# Run SMAC BlackBoxFacade on certain BBOB problem
python smacbenchmarking/run.py +optimizer/smac20=blackbox +problem/BBOB=cfg_4_1_4_0 seed=1

# Run SMAC BlackBoxFacade on all available BBOB problems for 10 seeds
python smacbenchmarking/run.py +optimizer/smac20=blackbox '+problem/BBOB=glob(*)' 'seed=range(1,11)'
```

## Containerization
To run benchmarking with containers, both the optimizer and benchmark have to be wrapped separately. 
We use Singularity/ Apptainer for this purpose.
The following example illustrates the principle based on a `DummyOptimizer` and `DummyBenchmark`.

### Optimizer
A Singularity recipe has to be created for the optimizer, which should be saved in the folder `container_recipes`.
This recipe has the purpose of setting up a container in which the optimizer can be run, e.g., installing the 
required packages, setting environment variables, copying files and so on.
For the `DummyOptimizer` this is `container_recipes/dummy_optimizer/dummy_optimizer.recipe`, which you can consult 
as a basis for other optimizers.

The optimizer then has to be built to an image named after the optimizer id, e.g., `DUMMY_Optimizer.sif` for the
`DummyOptimizer` using the following command:

```bash
singularity build DUMMY_optimizer.sif container_recipes/DUMMY_Optimizer/DUMMY_Optimizer.recipe
```

To facilitate this process, a short script is provided for this purpose, which is however system-specific to Noctua2.
It can be run as follows:

```bash
./compile_noctua2.sh build DUMMY_optimizer.sif container_recipes/DUMMY_Optimizer/DUMMY_Optimizer.recipe
```

### Benchmark
Like for the optimizer, a Singularity recipe has to be created for the benchmark, which should be saved in the folder
`container_recipes` as well.

The benchmark image also has to be according to the benchmark id, e.g., `DUMMY_Problem.sif` for the 
`DummyBenchmark` 
using
the following command:

```bash
singularity build DUMMY_benchmark.sif container_recipes/DUMMY_Problem/DUMMY_Problem.recipe
```

### Running
A third container is needed that handles the hydra config. It does not need to be adjusted for each optimizer or
benchmark, but can be used as is. It can be built as follows:

```bash
singularity build hydra_initializer.sif container_recipes/hydra_initializer.recipe
```

Running the containerized benchmarking system is also system-dependent. An example for Noctua2 is provided in the
script `start_container_noctua2.sh`. It can be run with hydra arguments, e.g. as follows:

```bash
./start_container_noctua2.sh +optimizer/DUMMY=config +problem/DUMMY=config
```

### Overview of the whole process

![Overview of the whole process](images/smac_benchmarking_containers.drawio.png)


The overall benchmarking system works as follows: 

We have three different containers wrapping different functionality and a shell script controlling these containers. 
The `HydraInitializer` container is responsible for constructing the Hydra configuration, 
which is required to initialize the `Optimizer` and the `Benchmark` container. 
The `Benchmark` container wraps the actual benchmark to be run and provides two main functionalities via a web service. 
First, it allows to get the search space associated with the benchmark. 
Second, it answers requests providing a configuration to be evaluated with the corresponding evaluation result.
The `Optimizer` container wraps the optimizer to be benchmarked and interacts with the `Benchmark` container.
Any information required to boot the containers is written to the hard drive by the `HydraInitializer` container. 

Note that we provide wrappers for the optimizer and the benchmark interfaces such that when you implement an 
optimizer or a benchmark within our benchmarking framework, 
you can ignore all aspects of the system just described and simply follow the simple API. 

## Open Questions

- How to aggregate and save data?
    - Performance data
        - trajectory: sorted by cost and time
- What metadata do we need?
    - General
        - Timestamp
        - Machine
    - Optimizer
        - Name
        - Repo
        - Commit
        - Version
    - SMACBenchmarking
        - Version
        - Commit

## Open Todos
- [ ] Containerize benchmarks / find solutions for requirements. Each optimizer could query a container during "run".
- [ ] Add budget under `task` as time AND number of function evaluations
    - If we use time as a budget, we need to check whether the hardware is the same.
- [ ] Create `dispatch.py` checking if run already exists
- [ ] Add slurm config
