import lightning as L
import hydra
from omegaconf import OmegaConf
import hydra.utils as hu
from ray.train.torch import TorchTrainer
from ray.train.lightning import prepare_trainer
from ray.train import ScalingConfig
from models.segformer.segformer import LitSegFormer  
from datamodule.goose_dataset import GooseDataset
from lightning.pytorch.utilities import rank_zero_only
from models.segformer.segformer import LitSegFormer  

@hydra.main(config_path="/home/autokarthik/autoware.off-road/ObjectSeg/configs", config_name="config", version_base="1.3")
def main(cfg):
    L.seed_everything(cfg.seed, workers=True)

    def train_loop(loop_cfg_dict):
        loop_cfg = OmegaConf.create(loop_cfg_dict)

        dm   = hu.instantiate(loop_cfg.datamodule)
        lit  = hu.instantiate(loop_cfg.model)
        

        callbacks = [hu.instantiate(c) for c in loop_cfg.callbacks.values()]
        logger    = None
        
        if rank_zero_only.rank == 0:
            logger = hu.instantiate(loop_cfg.logger)
            logger.log_hyperparams(OmegaConf.to_container(loop_cfg, resolve=True))
            logger.log_graph(lit)

        trainer = hu.instantiate(loop_cfg.trainer,
                                 callbacks=callbacks,
                                 logger=logger)
        trainer = prepare_trainer(trainer)
        trainer.fit(lit, dm)

    scaling_cfg = hu.instantiate(cfg.ray)         # ScalingConfig from YAML
    TorchTrainer(
        train_loop,
        scaling_config=scaling_cfg,
        train_loop_config=OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    ).fit()

if __name__ == "__main__":
    main()
