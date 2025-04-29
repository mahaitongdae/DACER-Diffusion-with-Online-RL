from gymnasium import register
import time

register(
    id='pusht-v0',
    entry_point='relax.env.pusht.pusht_env:PushTEnv',
    max_episode_steps=300
)

register(
    id='pusht-v1',
    entry_point='relax.env.pusht.pusht_shp_env:PushTShpEnv',
    max_episode_steps=300
)

register(
        id='pushtcurriculum-v0',
        entry_point='relax.env.pusht.pusht_env:PushTCurriculumEnv',
        max_episode_steps=300
    )

register(
        id='pushtcurriculum-v1',
        entry_point='relax.env.pusht.pusht_shp_env:PushTShpCurriculumEnv',
        max_episode_steps=300
    )

register(
        id='pushtcurriculum-v2',
        entry_point='relax.env.pusht.pusht_shp_env:PushTShpCurriculumRandomGoalEnv',
        max_episode_steps=300
    )
