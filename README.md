# SMACBenchmarking



## Installation
```
git clone https://github.com/AutoML/SMACBenchmarking.git
cd SMACBenchmarking
conda create -n smacbenchmarking python=3.11
conda activate smacbenchmarking

# Install for usage
pip install .

# Install for development
make install-dev
```

Documentation at https://AutoML.github.io/SMACBenchmarking/main

## Minimal Example

```bash
python smacbenchmarking/run.py +problem/BBOB=cfg_4_1_4_0 +optimizer/smac20=blackbox
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
