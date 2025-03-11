from gymnasium import register
import time

register(
    id='pusht-v0',
    entry_point='relax.env.pusht.pusht_env:PushTEnv',
    max_episode_steps=300
)
