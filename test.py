import tqdm

from tests.test_models import test_policy, test_policy_rollout, test_rollout
from tests.test_models import test_prior, test_prior_sample
from tests.test_baselines import test_bernoulli_ts

if __name__ == "__main__":
    tests = [
        test_policy,
        test_policy_rollout,
        test_rollout,
        # test_prior,
        # test_prior_sample,
        # test_bernoulli_ts
    ]
    for test in tqdm.tqdm(tests):
        test()
    print("All tests passed.")
