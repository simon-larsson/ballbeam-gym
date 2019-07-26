""" 
Base Environments

BallBeamBaseEnv - Base for all ball & beam environments

VisualBallBeamBase - Base for all environment that uses the simulation visualization 
                     as observation state 
"""

import time
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from ballbeam_gym.ballbeam import BallBeam

class BallBeamBaseEnv(gym.Env, EzPickle):
    """ BallBeamBaseEnv

    Base for all ball & beam environments

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    action_mode : action space, str ['continuous', 'discrete']
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, timestep=0.1, beam_length=1.0, max_angle=0.2, 
                 init_velocity=0.0, action_mode='continuous'):
        
        self.timestep = timestep
        self.beam_length = beam_length
        self.max_angle = max_angle
        self.action_mode = action_mode

        if action_mode == 'continuous':
            self.action_space = spaces.Box(low=np.array([-max_angle]), 
                                           high=np.array([max_angle]))
        elif action_mode == 'discrete':
            self.action_space = spaces.Discrete(2)
            self.angle = 0.0
            self.angle_change_speed = 0.8
        else:
            raise ValueError('Undefined action mode: `{}`'.format(action_mode))

        if init_velocity is None:
            self.random_init_velocity = True
            self.init_velocity = np.random.rand()
        else:
            self.random_init_velocity = False
            self.init_velocity = init_velocity  

        self.bb = BallBeam(timestep=self.timestep,
                           beam_length=beam_length,
                           max_angle=max_angle,
                           init_velocity=self.init_velocity)

        self.last_sleep = time.time()

    def _sleep_timestep(self):
        """ 
        Sleep to sync cycles to one timestep for rendering by 
        removing time spent on processing.
        """
        duration = time.time() - self.last_sleep
        if not duration > self.timestep:
            time.sleep(self.timestep - duration)
        self.last_sleep = time.time()

    def reset(self):
        """ 
        Reset simulation
        """
        if self.random_init_velocity is None:
            self.init_velocity = np.random.rand()
            self.bb.init_velocity = self.init_velocity

        self.bb.reset()

    def render(self, mode='human', close=False):
        """
        Render a timestep and sleep correct time

        Parameters
        ----------
        mode : rendering mode, str ['human']

        close : not used, bool
        """
        if mode == 'human':
            self.bb.render()
            self._sleep_timestep()

    def seed(self, seed):
        """
        Make environment deterministic

        Parameters
        ----------
        seed : seed number, int
        """
        np.random.seed(seed)

    def _action_conversion(self, action):
        """ 
        Convert action to proper domain action space (continuous)

        Parameters
        ----------
        action [continuous] : set angle, float (rad)
        action [discrete] : increase/descrease angle, int [0, 1]

        Returns
        -------
        action : set angle, float (rad)
        """
        if self.action_mode == 'discrete':
            if action == 0:
                self.angle += self.angle_change_speed*self.timestep
            else:
                self.angle -= self.angle_change_speed*self.timestep
            
            # limit min/max
            self.angle = max(-self.max_angle, min(self.max_angle, self.angle))
            action = self.angle

        return action

class VisualBallBeamBaseEnv(BallBeamBaseEnv):
    """ BallBeamBaseEnv

    Base for all ball & beam environments

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    action_mode : action space, str ['continuous', 'discrete']
    """

    metadata = {'render.modes': ['human', 'machine']}

    def __init__(self, timestep=0.1, beam_length=1.0, max_angle=0.2, 
                 init_velocity=0.0, action_mode='continuous', setpoint=None):

        kwargs = {'timestep': timestep,
                'beam_length': beam_length,
                'max_angle':max_angle, 
                'action_mode':action_mode, 
                'init_velocity': init_velocity}

        super().__init__(**kwargs)

        self.image_shape = (200, 250, 3)
        self.observation_space = spaces.Box(low=0, high=255.0, shape=(self.image_shape))

    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (image)
        """
        super.reset()
        return self._get_state()

    def render(self, mode='human', close=False):
        """
        Render a timestep and sleep correct time

        Parameters
        ----------
        mode : rendering mode, str ['human', 'machine']

        close : not used, bool
        """
        super.render(**{'mode': mode, 'close': close})

    def _get_state(self):
        """
        Get current simulation state from the plot

        Returns
        -------
        observation : simulation state, np.ndarray (image)
        """
        # update plot to get image
        self.bb.render(mode='machine')
        # take image from matplotlib and crop it to 200x250x3
        return np.array(self.bb.fig.canvas.renderer._renderer)[25:-25,130:390,:-1]

