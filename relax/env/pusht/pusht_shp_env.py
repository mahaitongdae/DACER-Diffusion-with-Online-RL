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
    masks = {
        'tee': [0.0, 0.5, 0.5, 0.0],
        'lee': [1.0, 0.0, 0.0, 0.0],
        'cee': [1.0, 0.0, 0.0, 1.0],
        # 'eye': [0.5, 0.5, 0.5, 0.5],
        # 'eff': [0.5, 0.0, 0.5, 0.0],
    }

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
            low=np.array([-8, -8, -8, -8, 0, 0, 0, 0, 0, 0,], dtype=np.float32),
            high=np.array([8, 8, 8, 8, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            shape=(10,),
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
        obs = np.concatenate([obs, self.masks[self.shape_type]])
        return obs.astype(np.float32)


class PushTShpCurriculumEnv(PushTCurriculumEnv):
    metadata = {"render.modes": [
        "human", "rgb_array"], "video.frames_per_second": 10}
    shapes = ['tee', 'cee', 'lee']
    masks = {
        'tee': [0.0, 0.5, 0.5, 0.0],
        'lee': [1.0, 0.0, 0.0, 0.0],
        'cee': [1.0, 0.0, 0.0, 1.0],
        'eye': [0.5, 0.5, 0.5, 0.5],
        'eff': [0.5, 0.0, 0.5, 0.0],
    }

    def __init__(self,
                 legacy = False,
                 block_cog = None, damping = None,
                 render_action = True,
                 render_size = 96,
                 reset_to_state = None,
                 ):
        super().__init__(legacy=legacy, block_cog=block_cog, damping=damping,
                         render_action=render_action, render_size=render_size,
                         reset_to_state=reset_to_state)

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
    
class PushTShpCurriculumRandomGoalEnv(PushTCurriculumEnv):
    metadata = {"render.modes": [
        "human", "rgb_array"], "video.frames_per_second": 10}
    shapes = ['tee', 'cee', 'lee']
    masks = {
        'tee': [0.0, 0.5, 0.5, 0.0],
        'lee': [1.0, 0.0, 0.0, 0.0],
        'cee': [1.0, 0.0, 0.0, 1.0],
        'eye': [0.5, 0.5, 0.5, 0.5],
        'eff': [0.5, 0.0, 0.5, 0.0],
    }

    def __init__(self,
                 legacy = False,
                 block_cog = None, damping = None,
                 render_action = True,
                 render_size = 96,
                 reset_to_state = None,
                 ):
        super().__init__(legacy=legacy, 
                         block_cog=block_cog, 
                         damping=damping,
                         render_action=render_action, 
                         render_size=render_size,
                         reset_to_state=reset_to_state,
                         random_goal_pose=True)

        self.observation_space = spaces.Box(
            low=np.array([-8, -8, -8, -8, 0, 0,-8, -8, 0, 0, 0, 0, 0, 0,], dtype=np.float32),
            high=np.array([8, 8, 8, 8, 1, 1, 8, 8, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            shape=(14,),
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
        obs = self.get_obs_rela_goal()
        obs = np.concatenate([obs, self.masks[self.shape_type]])
        return obs.astype(np.float32)
    
    def get_obs_rela_goal(self):
        obs = np.array(
            tuple(self.agent.position / self.window_size)
            + tuple(self.block.position / self.window_size)
            + (np.sin(self.block.angle), np.cos(self.block.angle))
            + tuple(self.goal_pose[0:2] / self.window_size)
            + (np.sin(self.goal_pose[-1]), np.cos(self.goal_pose[-1])))
        return obs.astype(np.float32)

        # based on block frame
        
        theta = self.goal_pose[-1]
        R_goal_from_world_inv = np.array([[np.cos(theta), -np.sin(theta)],
                                      [np.sin(theta), np.cos(theta)]])
        t_goal_from_world = np.array([self.goal_pose[0], self.goal_pose[1]]).reshape((2, 1))
        def world_to_local(pos_world):
            pos_world = pos_world.reshape((2, 1))
            pos_goal_local = R_goal_from_world_inv @ pos_world - R_goal_from_world_inv @ t_goal_from_world
            return pos_goal_local.flatten()
        
        agent_rel_pos = world_to_local(np.array([self.agent.position[0], self.agent.position[1]]))
        block_rel_pos = world_to_local(np.array([self.block.position[0], self.block.position[1]]))
        blk_real_angle = self.block.angle - self.goal_pose[2]
        obs = np.array([agent_rel_pos[0] / self.rela_pos_scale, agent_rel_pos[1] / self.rela_pos_scale,
                        block_rel_pos[0] / self.rela_pos_scale, block_rel_pos[1] / self.rela_pos_scale,
                        np.sin(blk_real_angle), np.cos(blk_real_angle)])
        return obs.astype(np.float32)

if __name__ == "__main__":
    from gymnasium import register
    import time

    register(
        id='pushtcurriculum-v1',
        entry_point='relax.env.pusht.pusht_shp_env:PushTShpCurriculumEnv',
        max_episode_steps=300
    )
    register(
        id='pusht-v1',
        entry_point='relax.env.pusht.pusht_shp_env:PushTShpEnv',
        max_episode_steps=300
    )

    register(
        id='pushtcurriculum-v2',
        entry_point='relax.env.pusht.pusht_shp_env:PushTShpCurriculumRandomGoalEnv',
        max_episode_steps=300
    )

    env = gym.make('pushtcurriculum-v2')
    env.reset(options={'shape_type': 'cee', 'curriculum_level': 0.1})
    for i in range(300):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        print(obs)
        # env.render()
        time.sleep(0.01)
