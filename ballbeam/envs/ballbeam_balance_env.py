import time
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from ballbeam.ballbeam import BallBeam

class BallBeamBalanceEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, time_step=0.1, beam_length=1.0, max_angle=0.2):
        self.time_step = time_step

        # angle
        self.action_space = spaces.Box(low=np.array([-max_angle]), high=np.array([max_angle]))
                                                       # [angle, position, velocity]
        self.observation_space = spaces.Box(low=np.array([-max_angle, -np.inf, -np.inf]), 
                                            high=np.array([max_angle, np.inf, np.inf]))

        self.bb = BallBeam(time_step=self.time_step,
                           beam_length=beam_length,
                           max_angle=max_angle,
                           init_velocity=np.random.rand())

    def step(self, action):
        self.bb.update(action)
        obs = np.array([self.bb.theta, self.bb.x, self.bb.v])
        reward = 1. if self.bb.on_beam else .0
        return obs, reward, not self.bb.on_beam, {}

    def reset(self):
        self.bb.reset()      
        return np.array([self.bb.theta, self.bb.x, self.bb.v])

    def render(self, mode='human', close=False):
        self.bb.render()
        time.sleep(self.time_step - 0.01)

    def seed(self, seed):
        np.random.seed(seed)
