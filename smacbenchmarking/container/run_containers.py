import os

import hydra
from domdf_python_tools.utils import printr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


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
    hydra_cfg = HydraConfig.instance().get()

    # write hydra config to file
    cfg_path = f"{hydra_cfg.run.dir}/hydra_config.yaml"
    print(cfg_path)
    OmegaConf.save(config=cfg, f=cfg_path)

    job_id = os.environ["SLURM_JOB_ID"]

    # write the cfg_path to file job_id_config.yaml
    with open(f"{job_id}_config.txt", 'w+') as f:
        f.write(cfg_path)

    with open(f"{job_id}_problem_container.txt", 'w+') as f:
        f.write(cfg_dict["benchmark_id"])

    with open(f"{job_id}_optimizer_container.txt", 'w+') as f:
        f.write(cfg_dict["optimizer_id"])

    return None


if __name__ == "__main__":
    main()
