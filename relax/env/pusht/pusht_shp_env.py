import gymnasium as gym
from gymnasium import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from relax.env.pusht.pusht_env import PushTEnv, PushTCurriculumEnv


class PushTShpEnv(PushTEnv):
    metadata = {"render.modes": [
        "human", "rgb_array"], "video.frames_per_second": 10}
    shapes = ['tee', 'cee', 'lee']

    def __init__(self,
                 legacy=False,
                 block_cog=None, damping=None,
                 render_action=True,
                 render_size=96,
                 reset_to_state=None,
                 ):
        super().__init__(legacy=legacy, block_cog=block_cog, damping=damping,
                         render_action=render_action, render_size=render_size,
                         reset_to_state=reset_to_state)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, len(self.shapes)], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if options is None or 'shape_type' not in options:
            if seed is None:
                seed = np.random.randint(0, 2**32-1)
            rng = np.random.default_rng(seed=seed)
            shape_type = self.shapes[rng.integers(0, len(self.shapes))]
        else:
            shape_type = options['shape_type']
        self.shape_type = shape_type
        options = {} if options is None else options
        options['shape_type'] = shape_type
        return super().reset(seed, options=options)

    def _get_obs(self):
        obs = super()._get_obs()
        obs = np.concatenate([obs, [self.shapes.index(self.shape_type)]])
        return obs.astype(np.float32)


class PushTShpCurriculumEnv(PushTCurriculumEnv):
    metadata = {"render.modes": [
        "human", "rgb_array"], "video.frames_per_second": 10}
    shapes = ['tee', 'cee', 'lee']
    masks = {
        'tee': [0.0, 0.5, 0.5, 0.0],
        'lee': [1.0, 0.0, 0.0, 0.0],
        'cee': [1.0, 0.0, 0.0, 1.0],
    }

    def __init__(self,
                 legacy = False,
                 block_cog = None, damping = None,
                 render_action = True,
                 render_size = 96,
                 reset_to_state = None,
                 curriculum_level=0.0,
                 render_mode=None,
                 ):
        super().__init__(legacy=legacy, block_cog=block_cog, damping=damping,
                         render_action=render_action, render_size=render_size,
                         reset_to_state=reset_to_state, curriculum_level=curriculum_level)

        self.observation_space = spaces.Box(
            low=np.array([-8, -8, -8, -8, 0, 0, 0, 0, 0, 0,], dtype=np.float32),
            high=np.array([8, 8, 8, 8, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            shape=(10,),
            dtype=np.float32
        )

    def reset(self, seed = None, options = None):
        if options is None or 'shape_type' not in options:
            if seed is None:
                seed = np.random.randint(0, 2 ** 32 - 1)
            rng = np.random.default_rng(seed=seed)
            shape_type = self.shapes[rng.integers(0, len(self.shapes))]
        else:
            shape_type = options['shape_type']
        self.shape_type = shape_type
        options = {} if options is None else options
        options['shape_type'] = shape_type
        return super().reset(seed, options=options)

    def _get_obs(self):
        obs = super()._get_obs()
        obs = np.concatenate([obs, self.masks[self.shape_type]])
        return obs.astype(np.float32)

if __name__ == "__main__":
    from gymnasium import register
    import time

    register(
        id='pushtcurriculum-v1',
        entry_point='relax.env.pusht.pusht_shp_env:PushTShpCurriculumEnv',
        max_episode_steps=300
    )

    env = gym.make('pushtcurriculum-v1')
    env.reset()
    for i in range(300):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        print(obs)
        env.render()
        time.sleep(0.01)
