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
### Database
All results will be written to a central database.
This database needs to be set up once on the server.
MySQL can be installed with the information [here](https://dev.mysql.com/doc/refman/8.0/en/linux-installation.html).


Documentation at https://AutoML.github.io/SMACBenchmarking/main

## Minimal Example

```bash
# Run SMAC BlackBoxFacade on certain BBOB problem
python smacbenchmarking/run.py +optimizer/smac20=blackbox +problem/BBOB=cfg_4_1_4_0 seed=1

# Run SMAC BlackBoxFacade on all available BBOB problems for 10 seeds
python smacbenchmarking/run.py +optimizer/smac20=blackbox '+problem/BBOB=glob(*)' 'seed=range(1,11)'
```


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
