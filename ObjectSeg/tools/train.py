import lightning as L
import hydra
from omegaconf import OmegaConf
import hydra.utils as hu
from ray.train.torch import TorchTrainer
from ray.train.lightning import prepare_trainer
from ray.train import ScalingConfig
from segformer_model import LitSegFormer   # wraps SegFormer + loss + metrics

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    L.seed_everything(cfg.seed, workers=True)

    def train_loop(loop_cfg_dict):
        loop_cfg = OmegaConf.create(loop_cfg_dict)

        dm   = hu.instantiate(loop_cfg.datamodule)
        net  = hu.instantiate(loop_cfg.model)
        lit  = LitSegFormer(model=net,
                            optimizer_cfg=loop_cfg.optim.optimizer,
                            scheduler_cfg=loop_cfg.optim.scheduler)

        callbacks = [hu.instantiate(c) for c in loop_cfg.callbacks]
        logger    = None
        if L.utilities.rank_zero.rank_zero_only.rank_zero():
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
        train_loop_config=OmegaConf.to_container(cfg, resolve=False)
    ).fit()

if __name__ == "__main__":
    main()
