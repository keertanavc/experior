import tqdm
import hydra

from omegaconf import OmegaConf
from src.configs import ExperiorConfig

from tests.test_models import (
    test_transformer_policy,
    test_transformer_policy_rollout,
    test_softelim_policy_rollout,
    test_rollout,
    test_softelim_policy,
    test_beta_ts_policy,
    test_prior,
    test_prior_sample,
)
from tests.test_posterior import test_sglangevin
from tests.test_baselines import test_bernoulli_ts
from tests.test_envs import test_bernoulli_bandit


@hydra.main(version_base=None, config_path="test_conf", config_name="config")
def main(conf: ExperiorConfig):
    conf = hydra.utils.instantiate(conf)
    conf = ExperiorConfig(**OmegaConf.to_container(conf))

    tests = [
        # test_beta_ts_policy,
        # test_softelim_policy,
        # test_transformer_policy,
        # test_transformer_policy_rollout,
        # test_softelim_policy_rollout,
        # test_rollout,
        # test_prior,
        # test_prior_sample,
        # test_bernoulli_ts,
        # test_sglangevin,
        test_bernoulli_bandit
    ]

    for test in tqdm.tqdm(tests):
        test(conf)


if __name__ == "__main__":
    main()
    print("All tests passed.")
