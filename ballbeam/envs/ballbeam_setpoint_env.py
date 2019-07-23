import time
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from ballbeam.ballbeam import BallBeam

class BallBeamSetpointEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, timestep=0.1, setpoint=None, beam_length=1.0, max_angle=0.2):
        self.timestep = timestep

        if setpoint is None:
            self.setpoint = np.random.random_sample()*beam_length - beam_length/2
            self.random_setpoint = True
        else:
            if abs(setpoint) > beam_length/2:
                raise ValueError('Setpoint outside of beam.')

            self.setpoint = setpoint
            self.random_setpoint = False

        self.beam_length = beam_length

        # angle
        self.action_space = spaces.Box(low=np.array([-max_angle]), high=np.array([max_angle]))
                                                    # [angle, position, velocity, setpoint]
        self.observation_space = spaces.Box(low=np.array([-max_angle, -np.inf, -np.inf, -beam_length/2]), 
                                            high=np.array([max_angle, np.inf, np.inf, beam_length/2]))

        self.bb = BallBeam(timestep=self.timestep,
                           max_angle=max_angle,
                           beam_length=beam_length)

    def step(self, action):
        self.bb.update(action)
        obs = np.array([self.bb.theta, self.bb.x, self.bb.v, self.setpoint])
        # reward squared proximity to setpoint 
        reward = (1 - (self.setpoint - self.bb.x)/self.bb.L)**2
        
        return obs, reward, not self.bb.on_beam, {}

    def reset(self):
        self.bb.reset()
        
        if self.random_setpoint is None:
            self.setpoint = np.random.random_sample()*self.beam_length \
                            - self.beam_length/2

        return np.array([self.bb.theta, self.bb.x, self.bb.v, self.setpoint])

    def render(self, mode='human', close=False):
        self.bb.render(self.setpoint)
        time.sleep(self.timestep - 0.01)

    def seed(self, seed):
        np.random.seed(seed)
