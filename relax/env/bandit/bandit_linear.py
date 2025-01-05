import numpy as np
import itertools
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import jax

'''
The environment of contextual linear bandit with continuous action space
'''
class CBandit(gym.Env):
  def __init__(self, action_dim=4, context_dim=4, noise_scale=0.03,
               generation_seed=0):
    self._action_dim = action_dim
    self._context_dim = context_dim
    self._noise_scale = noise_scale
    self._generate_bandit(generation_seed)
    self._iteration = 0
    self._total_optimal_reward = 0

    self.observation_space = spaces.Box(low=-1, high=1, shape=(self._context_dim,), dtype=np.float64)
    self.action_space = spaces.Box(low=-1, high=1, shape=(self._action_dim,), dtype=np.float64)
    
    self.seed()
    self.reset()

  def _generate_bandit(self, seed):
    # gen_random, _ = seeding.np_random(seed)
    # self._coefficients = gen_random.random(self._action_dim) * 2 - 1 
    self._coefficients = np.array([-0.35507606, -0.48329438, -0.25937637,  0.77009931])
    print("Linear bandit coefficients:", self._coefficients)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self, seed=None, options=None):
    self.state = self._get_obs(seed)
    return self.state, {}

  def _get_obs(self, seed=None):
    gen_random, _ = seeding.np_random(seed)
    return gen_random.random((self._context_dim, )) * 2 - 1

  def step(self, action):
    obs = self.state
    phi_output = 2 * obs * action
    reward = np.sum(phi_output * self._coefficients)

    # noise = self.np_random.normal(loc=0.0, scale=self._noise_scale)  
    # sampled_reward = reward + noise
    sampled_reward = reward

    # all_actions = list(itertools.product([-1, 1], repeat=4))  
    # optimal_reward = -1000000
    # for a in all_actions:
    #   a = np.array(a)
    #   phi_temp = 2 * obs * a
    #   reward_temp = np.sum(phi_temp * self._coefficients)
    #   optimal_reward = max(optimal_reward, reward_temp)
    # self._total_optimal_reward += optimal_reward
    # self._iteration += 1
    # if self._iteration == 10000:
    #   print("optimal reward")
    #   print(str(self._total_optimal_reward / self._iteration))
    #   self._iteration = 0
    #   self._total_optimal_reward = 0
    
      
    done = True
    return obs, sampled_reward, done, False, {}
  

