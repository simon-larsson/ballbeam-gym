import time
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from ballbeam.ballbeam import BallBeam

class BallBeamSetpointEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, time_step=0.1, setpoint=0.4, max_angle=0.2):
        self.time_step = time_step
        self.setpoint = setpoint
        # angle
        self.action_space = spaces.Box(low=np.array([-max_angle]), high=np.array([max_angle]))
                                                    # [angle, position, velocity, setpoint]
        self.observation_space = spaces.Box(low=np.array([-max_angle, -np.inf, -np.inf, -0.5]), 
                                            high=np.array([max_angle, np.inf, np.inf, 0.5]))

        self.bb = BallBeam(time_step=self.time_step)

    def step(self, action):
        self.bb.update(action)
        obs = np.array([self.bb.theta, self.bb.x, self.bb.v, self.setpoint])
        # reward squared proximity to setpoint 
        reward = (1 - (self.setpoint - self.bb.x)/self.bb.L)**2
        
        return obs, reward, not self.bb.on_beam, {}

    def reset(self):
        self.bb.reset()      
        return np.array([self.bb.theta, self.bb.x, self.bb.v, self.setpoint])

    def render(self, mode='human', close=False):
        self.bb.render(self.setpoint)
        time.sleep(self.time_step - 0.01)

    def seed(self, seed):
        np.random.seed(seed)
