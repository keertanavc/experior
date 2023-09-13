import hydra
import wandb

from omegaconf import OmegaConf

from src.trainers import MiniMaxTrainer, MaxEntTrainer
from src.utils import PRNGSequence, init_run_dir
from src.configs import ExperiorConfig

from pprint import pprint

from jax import config

config.update("jax_debug_nans", True)

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: ExperiorConfig):
    conf = hydra.utils.instantiate(conf)
    conf = ExperiorConfig(**OmegaConf.to_container(conf))

    if conf.test_run:
        pprint(conf.dict())
    else:
        conf = init_run_dir(conf)
        wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=conf.dict(),
            name=conf.wandb.run_name,
            id=conf.wandb.run_id,
            resume="allow" if conf.wandb.resume else False,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
        )
        wandb.define_metric("prior/step")
        wandb.define_metric("policy/step")
        wandb.define_metric("prior/*", step_metric="prior/step")
        wandb.define_metric("policy/*", step_metric="policy/step")

    rng = PRNGSequence(conf.seed)
    if conf.trainer.name == "minimax":
        trainer = MiniMaxTrainer(conf)
    elif conf.trainer.name == "MaxEnt":
        trainer = MaxEntTrainer(conf)

    trainer.initialize(next(rng))
    trainer.train(next(rng))

    if not conf.test_run:
        trainer.save_metrics(next(rng))


if __name__ == "__main__":
    main()
