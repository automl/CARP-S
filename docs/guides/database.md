# Database

This document describes how to set up the database for the CARP-S framework and use it for
logging experiment results and trajectories.

Either SQLite or MySQL can be used as database, which has some slight differences. 
Using SQLite is straightforward; you get a local database file but
parallel execution is not efficient at all. You configure the used database in the 
`carps/experimenter/py_experimenter.yaml` file by changing the `provider` to `mysql` or 
`sqlite`. 

In any case, before you can start any jobs, the jobs need to be dispatched to the database.
To this end, call the file `create_cluster_configs.py` with the desired hydra arguments.
This can be done locally or on the server if you can execute python there directly.
If you execute it locally, the database file `carps.db` will be created in the current directory and 
needs to be transferred to the cluster.

```bash
python carps/container/create_cluster_configs.py +optimizer/DUMMY=config +task/DUMMY=config 'seed=range(1,21)' --multirun
```

If you want to use a personal/local MySQL database, follow these steps:

1. Setup MySQL ([tutorial](https://dev.mysql.com/doc/refman/8.3/en/installing.html))


2. Create database via `mysql> CREATE DATABASE carps;`
    Select password as authentification.
    Per default, the database name is `carps`.
    It is set in `carps/experimenter/py_experimenter.yaml`.


3. Add credential file at `carps/experimenter/credentials.yaml`, e.g.
```yaml
CREDENTIALS:
  Database:
    user: root
    password: <password>
  Connection:
    Standard:
      server: localhost
```


4. Set flag not to use ssh server in `carps/experimenter/py_experimenter.yaml` if you are on your local machine.

