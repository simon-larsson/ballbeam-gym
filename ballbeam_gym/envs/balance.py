""" 
Balance Environments

Environments where the objective is to keep the ball on the beam

BallBeamBalanceEnv - Balance environment with a state consisting of key state variables

VisualBallBeamBalanceEnv - Balance environment with simulation plot as state
"""

import numpy as np
from gym import spaces
from ballbeam_gym.envs.base import BallBeamBaseEnv, VisualBallBeamBaseEnv

class BallBeamBalanceEnv(BallBeamBaseEnv):
    """ BallBeamBalanceEnv

    Balance environment with a state consisting of key variables

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

        # reward is 1 every time ball stays on beam
        reward = 1.0 if self.bb.on_beam else 0.0
        
        return obs, reward, not self.bb.on_beam, {}

    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        super().reset()
        return np.array([self.bb.theta, self.bb.x, self.bb.v])

class VisualBallBeamBalanceEnv(VisualBallBeamBaseEnv):
    """ VisualBallBeamBalanceEnv

    Balance environment with simulation plot as state

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

        # reward is 1 every time ball stays on beam
        reward = 1.0 if self.bb.on_beam else 0.0

        return obs, reward, not self.bb.on_beam, {}


