# Commands

You can run a certain problem and optimizer combination directly with Hydra via:
```bash
python -m carps.run +problem=... +optimizer=... seed=... -m
```

Another option is to fill the database with all possible combinations of problems and optimizers
you would like to run:
```bash
carps --create_cluster_configs --optimizer DUMMY/config --problem DUMMY/config
```

Then, run them from the database with:
```bash
python -m carps.run_from_db 
```

To check whether any runs are missing, you can use the following command. It will create
a file `runcommands_missing.sh` containing the missing runs:
```bash
python -m carps.utils.check_missing rundir
```

To gather the logs from the files, you can use the following command:
```bash
python -m carps.analysis.gather_data rundir
```