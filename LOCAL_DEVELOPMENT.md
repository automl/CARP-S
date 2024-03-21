# NoContainer Mode


## Virtual Environments
You can try to install all dependencies into one big environment, but probably there are package clashes.
Therefore, you can build one virtual environment for each optimizer-benchmark combination. Either run `scripts/build_envs.sh` to build all
existing combinations or copy the combination and run as needed. It will create an environment with name `automlsuite_${OPTIMIZER_CONTAINER_ID}_${BENCHMARK_ID}`.

## Running Parallel
You can run your optimization via
```bash
conda run -n automlsuite_DUMMY_Optimizer_DUMMY_Problem python smacbenchmarking/run.py +optimizer/DUMMY=config +problem/DUMMY=config  'seed=range(1,11)' +cluster=local -m
```
This uses joblib parallelization on your local machine. If you are on a slurm cluster, you can specify `+cluster=slurm` and adapt 
this to your needs. Check [this page](for more launchers)[https://hydra.cc/docs/plugins/joblib_launcher/], e.g. Ray or RQ besides Joblib and Submitit.

# Container Mode
## Database
If you want to use containers and a personal/local database, follow these steps:

1. Setup MySQL ([tutorial](https://dev.mysql.com/doc/refman/8.3/en/installing.html))
2. Create database via `mysql> CREATE DATABASE smacbenchmarking;` 
Select password as authentification. Per default, the database name is `smacbenchmarking`. It is set in `smacbenchmarking/container/py_experimenter.yaml`.
2. Add credential file at `smacbenchmarking/container/credentials.yaml`, e.g.
```yaml
CREDENTIALS:
  Database:
    user: root
    password: <password>
  Connection:
    Standard:
      server: localhost
```
3. Set flag not to use ssh server in `smacbenchmarking/container/py_experimenter.yaml` if you are on your local machine.
