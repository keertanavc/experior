class TransformerPolicyConfig:
    def __init__(self, horizon, num_actions, n_blocks, h_dim, num_heads, drop_p, dtype):
        self.horizon = horizon
        self.num_actions = num_actions
        self.n_blocks = n_blocks
        self.h_dim = h_dim
        self.drop_p = drop_p
        self.num_heads = num_heads
        self.dtype = dtype


class OptimizerConfig:
    def __init__(self, policy_lr, prior_lr, mc_samples):
        self.policy_lr = policy_lr
        self.prior_lr = prior_lr
        self.mc_samples = mc_samples


class GlobalConfig:
    def __init__(self, seed, epochs):
        self.seed = seed
        self.epochs = epochs
