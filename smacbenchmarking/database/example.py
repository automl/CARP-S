from __future__ import annotations

from rich import print as printr
from smacbenchmarking.database import utils
from py_experimenter.result_processor import ResultProcessor
import logging
from smacbenchmarking.database.utils import load_config
from smacbenchmarking.database.connector import DatabaseConnectorMySQL
from smacbenchmarking.database.result_processor import ResultProcessor
from omegaconf import OmegaConf


logging.basicConfig(level=logging.DEBUG)

# Load config
config_fn = "smacbenchmarking/configs/database.yaml"
cfg = load_config(config_fn)
problem_cfg = load_config("/home/numina/Documents/repos/SMACBenchmarking/smacbenchmarking/configs/problem/BBOB/cfg_4_1_4_0.yaml")
optimizer_cfg = load_config("/home/numina/Documents/repos/SMACBenchmarking/smacbenchmarking/configs/optimizer/smac20/blackbox.yaml")
configs = (cfg, problem_cfg, optimizer_cfg)
cfg = OmegaConf.merge(*configs)
printr(cfg)


# Register experiment
connector = DatabaseConnectorMySQL(database_cfg=cfg.database)
parameters = utils.get_keyfield_data(cfg)
printr(parameters)

connector.config = cfg
connector.fill_table(parameters=parameters)

# Get experiment id
experiment_id = connector.find_experiment_id(parameters)
printr(experiment_id)




# TODO Should we save additional info?


# Process results belonging to experiment id
table_name = cfg.database.table_name
result_processor = ResultProcessor(
    config=cfg, 
    use_codecarbon=False, 
    table_name=table_name,
    codecarbon_config=None,
    database_cfg=cfg.database,
    experiment_id=experiment_id,  # The AUTO_INCREMENT is 1-based
    result_fields=cfg["database"]["resultfields"],
    )
print("Created tables")


log = {
    "trials": {
        "n_trials": 3,
        "trial_info__config": str([0,1]),
        "trial_info__instance": 0,
        "trial_info__seed": 0,
        "trial_info__budget": 0,
        "trial_value__cost": str(100),
        "trial_value__time": 3,
        "trial_value__starttime": 12345,
        "trial_value__starttime": 12348,
        "trial_value__status": "OK",
    }
}
result_processor.process_logs(log)



# # Connect to the database
# connection = pymysql.connect(host='localhost',
#                              user='root',
#                              password='samtigerpilz',
#                              database='db',
#                              cursorclass=pymysql.cursors.DictCursor)

# with connection:
#     with connection.cursor() as cursor:
#         # Create a new record
#         sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
#         cursor.execute(sql, ('webmaster@python.org', 'very-secret'))

#     # connection is not autocommit by default. So you must commit to save
#     # your changes.
#     connection.commit()

#     with connection.cursor() as cursor:
#         # Read a single record
#         sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
#         cursor.execute(sql, ('webmaster@python.org',))
#         result = cursor.fetchone()
#         print(result)