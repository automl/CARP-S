from domdf_python_tools.utils import printr
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from rich import inspect

from smacbenchmarking.run import make_problem, make_optimizer, save_run
from smacbenchmarking.utils.exceptions import NotSupportedError


@hydra.main(config_path="configs", config_name="base.yaml", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run optimizer on problem.

    Save trajectory and metadata to database.

    Parameters
    ----------
    cfg : DictConfig
        Global configuration.

    """
    # TODO: adapt to container structure: start problem container, start optimizer container, run optimizer, save run,
    #  stop problem (optimizer should stop itself)
    cfg_dict = OmegaConf.to_container(cfg=cfg, resolve=True)
    printr(cfg_dict)
    hydra_cfg = HydraConfig.instance().get()
    printr(hydra_cfg.run.dir)

    problem = make_problem(cfg=cfg)
    inspect(problem)

    optimizer = make_optimizer(cfg=cfg, problem=problem)
    inspect(optimizer)

    try:
        optimizer.run()
    except NotSupportedError:
        print("Not supported. Skipping.")
    except Exception as e:
        print("Something went wrong:")
        print(e)

    metadata = {"hi": "hello"}  # TODO add reasonable meta data

    save_run(cfg=cfg, optimizer=optimizer, metadata=metadata)

    return None


if __name__ == "__main__":
    main()
