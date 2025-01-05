from gymnasium.envs.registration import register

register(
    id="LinearBandit-v0",
    entry_point="bandit.bandit_linear:CBandit",
)