""" 
Throw Environments

Environments where the objective is to keep to throw the ball as far as possible

BallBeamThrowEnv - Throw environment with a state consisting of key state variables

VisualBallBeamThrowEnv - Throw environment with simulation plot as state
"""

from math import sqrt, atan, sin
import numpy as np
from gym import spaces
from ballbeam_gym.envs.base import BallBeamBaseEnv, VisualBallBeamBaseEnv

class BallBeamThrowEnv(BallBeamBaseEnv):
    """ BallBeamThrowEnv

    Throw environment with a state consisting of key variables

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    action_mode : action space, str ['continuous', 'discrete']
    """

    def __init__(self, timestep=0.1, beam_length=1.0, max_angle=0.2, 
                 init_velocity=None, action_mode='continuous'):
                 
        kwargs = {'timestep': timestep,
                  'beam_length': beam_length,
                  'max_angle':max_angle, 
                  'action_mode':action_mode, 
                  'init_velocity': init_velocity}

        super().__init__(**kwargs)
                                                        # [angle, position, velocity] 
        self.observation_space = spaces.Box(low=np.array([-max_angle, -np.inf, -np.inf]), 
                                            high=np.array([max_angle, np.inf, np.inf]))

        self.left_beam = False

    def step(self, action):
        """
        Update environment for one action

        Parameters
        ----------
        action [continuous] : set angle, float (rad)
        action [discrete] : increase/descrease angle, int [0, 1]
        """
        self.bb.update(self._action_conversion(action))
        obs = np.array([self.bb.theta, self.bb.x, self.bb.v])

        reward = 0

        done = self.left_beam

        if not self.bb.on_beam and not self.left_beam:
            
            v0 = sqrt(self.bb.v_y**2 + self.bb.v_x**2)
            angle = atan(self.bb.v_y / self.bb.v_x)
            distance = (v0**2/self.bb.g)*(1 + sqrt(1 + (2*self.bb.g*self.bb.y)/v0**2*sin(angle)**2))*sin(2*angle) - self.bb.r
            reward = max(reward, distance)    
            self.left_beam = True
        
        return obs, reward, done, {}

    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        super().reset()
        self.left_beam = False
        return np.array([self.bb.theta, self.bb.x, self.bb.v])

class VisualBallBeamThrowEnv(VisualBallBeamBaseEnv):
    """ VisualBallBeamThrowEnv

    Throw environment with simulation plot as state

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    action_mode : action space, str ['continuous', 'discrete']
    """

    def __init__(self, timestep=0.1, beam_length=1.0, max_angle=0.2, 
                 init_velocity=None, action_mode='continuous'):

        kwargs = {'timestep': timestep,
                  'beam_length': beam_length,
                  'max_angle':max_angle, 
                  'action_mode':action_mode, 
                  'init_velocity': init_velocity}

        super().__init__(**kwargs)

        self.left_beam = False

    def step(self, action):
        """
        Update environment for one action

        Parameters
        ----------
        action [continuous] : set angle, float (rad)
        action [discrete] : increase/descrease angle, int [0, 1]
        """
        self.bb.update(self._action_conversion(action))
        obs = self._get_state()

        reward = 0

        done = self.left_beam

        if not self.bb.on_beam and not self.left_beam:
            
            v0 = sqrt(self.bb.v_y**2 + self.bb.v_x**2)
            angle = atan(self.bb.v_y / self.bb.v_x)    
            distance = (v0**2/self.bb.g)*(1 + sqrt(1 + (2*self.bb.g*self.bb.y)/v0**2*sin(angle)**2))*sin(2*angle) - self.bb.r
            reward = max(reward, distance)    
            self.left_beam = True

        return obs, reward, done, {}

    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        
        self.left_beam = False
        return super().reset()
