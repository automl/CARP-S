# v0.1.2

## Benchmarks
- HPOBench: Add rl, nasbench_101, nasbench_201, nasbench1shot1, and nas_hpo (#155, fix for benchmarks #158)
- Pymoo: Add all unconstraint problems (#162)

## Optimizers
- Add Ax as optimizer (#166)

## Improvements
- Update/clean optimizer configs, update HEBO+Skopt a bit (#161, #163)
- Add BBOB-Vizier setup (but using BBOB from ioh; d=20, n_trials=100, #163)

# v0.1.1

## Documentation
- Add example for the case when developing a standalone package either for a benchmark or for an optimizer (#157)

## Bug Fixes
- Disallowed HPOBench use in carp-s container, changed HPOBench mode to container mode (#94).
- Fix HEBO integration (#149).
- Fix plotting and data gathering

# v0.1.0

- Initial version of CARP-S.