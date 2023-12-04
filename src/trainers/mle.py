import jax
import chex

from src.configs import ExperiorConfig
from src.experts import Expert
from src.losses import prior_mle_loss
from .maxent import MaxEntTrainer


class MLETrainer(MaxEntTrainer):
    def __init__(self, conf: ExperiorConfig, expert: Expert):
        super().__init__(conf, expert)

    def initialize(self, rng: chex.PRNGKey):
        super().initialize(rng)

        @jax.jit
        def _policy_step(key, policy_state, prior_state, batch):
            def policy_loss_fn(policy_params):
                policy_prior_log_p = policy_state.apply_fn(
                    {"params": policy_params}, batch, method="prior_log_p"
                )

                density = prior_state.apply_fn({"params": prior_state.params}, batch)
                loss = prior_mle_loss(batch, policy_prior_log_p, density)

                out = {}
                out["loss"] = loss
                return loss, out

            (loss, out), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(policy_state.params)

            policy_state = policy_state.apply_gradients(grads=policy_grads)

            return policy_state, out

        self._policy_step = _policy_step
