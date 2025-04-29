from gymnasium import register
import time

register(
    id='pusht-v0',
    entry_point='relax.env.pusht.pusht_env:PushTEnv',
    max_episode_steps=300
)

# no curriculum
register(
    id='pusht-v1',
    entry_point='relax.env.pusht.pusht_shp_env:PushTShpEnv',
    max_episode_steps=300
)

# no shape
register(
        id='pushtcurriculum-v0',
        entry_point='relax.env.pusht.pusht_env:PushTCurriculumEnv',
        max_episode_steps=300
    )

# legacy
register(
        id='pushtcurriculum-v1',
        entry_point='relax.env.pusht.pusht_shp_env:PushTShpCurriculumEnv',
        max_episode_steps=300
    )

# Random goal
register(
        id='pushtcurriculum-v2',
        entry_point='relax.env.pusht.pusht_shp_env:PushTShpCurriculumRandomGoalEnv',
        max_episode_steps=300
    )

# no reward shaping
register(
        id='pushtcurriculum_sparse-v0',
        entry_point='relax.env.pusht.pusht_ablation_env:PushTShpSparseCurriculumEnv',
        max_episode_steps=300
    )