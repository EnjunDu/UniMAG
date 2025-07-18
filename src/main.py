# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.task.name == "nc":
        from graph_centric.nc.run_batch import run_nc
        run_nc(cfg)
    elif cfg.task.name == "lp":
        from graph_centric.lp.run_batch import run_lp
        run_lp(cfg)
    elif cfg.task.name == "gc":
        from graph_centric.gc.run import run_gc
        run_gc(cfg)
    elif cfg.task.name in ["modality_matching", "modality_retrieval", "modality_alignment"]:
        from multimodal_centric.qe.run import run_qe
        run_qe(cfg)

if __name__ == "__main__":
    main()
