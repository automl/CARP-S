import hydra
from domdf_python_tools.utils import printr
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

    # write hydra config to file
    cfg_path = f"cluster_configs/{cfg.__hash__()}.yaml"
    print(cfg_path)
    OmegaConf.save(config=cfg, f=cfg_path)
    return None


if __name__ == "__main__":
    main()

