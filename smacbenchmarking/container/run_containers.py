import hydra
from domdf_python_tools.utils import printr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich import inspect
from spython.main import Client

from smacbenchmarking.run import make_optimizer, make_problem, save_run
from smacbenchmarking.utils.exceptions import NotSupportedError


@hydra.main(config_path="../configs", config_name="base.yaml", version_base=None)  # type: ignore[misc]
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

    # write hydra config to file
    print(f"{hydra_cfg.run.dir}/hydra_config.yaml")
    OmegaConf.save(config=cfg, f=f"{hydra_cfg.run.dir}/hydra_config.yaml")

    image_name = cfg_dict["benchmark_id"]
    problem_instance = Client.instance(f"{image_name}.sif")
    problem_instance.run()

    # problem = make_problem(cfg=cfg)
    # inspect(problem)

    # optimizer = make_optimizer(cfg=cfg, problem=problem)
    # inspect(optimizer)

    # try:
    #    optimizer.run()
    # except NotSupportedError:
    #    print("Not supported. Skipping.")
    # except Exception as e:
    #    print("Something went wrong:")
    #    print(e)

    # metadata = {"hi": "hello"}  # TODO add reasonable meta data

    # save_run(cfg=cfg, optimizer=optimizer, metadata=metadata)

    return None


if __name__ == "__main__":
    main()
