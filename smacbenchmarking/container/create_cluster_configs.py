import hydra
from domdf_python_tools.utils import printr
from omegaconf import DictConfig, OmegaConf
from py_experimenter.experimenter import PyExperimenter


@hydra.main(config_path="../configs", config_name="base.yaml", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run optimizer on problem.

    Save trajectory and metadata to database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.

    """
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)

    experiment_configuration_file_path = 'smacbenchmarking/container/py_experimenter.cfg'
    experimenter = PyExperimenter(experiment_configuration_file_path=experiment_configuration_file_path,
                                  name='smacbenchmarking')

    rows = [{
        'config': cfg_dict,
        'problem_id': cfg_dict["benchmark_id"],
        'optimizer_id': cfg_dict["optimizer_id"],
    }]
    experimenter.fill_table_with_rows(rows)
    return None


if __name__ == "__main__":
    main()