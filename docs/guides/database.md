# Database

Per default, `carps` logs to files. This has its caveats: Checking experiment status is a bit more cumbersome (but 
possible with `python -m carps.utils.check_missing <rundir>` to check for missing/failed experiments) and reading from
the filesystem takes a long time. For this reason, we can also control and log experiments to a MySQL database with
`PyExperimenter`.

This document describes how to set up the database for the CARP-S framework and use it for
logging experiment results and trajectories.

Either SQLite or MySQL can be used as database, which has some slight differences. 
Using SQLite is straightforward; you get a local database file but
parallel execution is not efficient at all. 

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

You configure the used database in the 
`carps/experimenter/py_experimenter.yaml` file by changing the `provider` to `mysql` or 
`sqlite`. 


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

This can be done locally or on the server if you can execute python there directly.
If you execute it locally, the database file `carps.db` will be created in the current directory and 
needs to be transferred to the cluster.
Example:
```bash
python carps/container/create_cluster_configs.py +optimizer/randomsearch=config +task/DUMMY=config 'seed=range(1,21)' -m
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