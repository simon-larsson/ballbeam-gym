""" 
Setpoint Environments

Environments where the objective is to keep the ball close to a set beam postion

BallBeamSetpointEnv - Setpoint environment with a state consisting of key variables

VisualBallBeamSetpointEnv - Setpoint environment with simulation plot as state
"""

import numpy as np
from gym import spaces
from ballbeam_gym.envs.base import BallBeamBaseEnv, VisualBallBeamBaseEnv

class BallBeamSetpointEnv(BallBeamBaseEnv):
    """ BallBeamSetpointEnv

    Setpoint environment with a state consisting of key variables

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    action_mode : action space, str ['continuous', 'discrete']

    setpoint : target position of ball, float (units)
    """

    def __init__(self, timestep=0.1, beam_length=1.0, max_angle=0.2, 
                 init_velocity=0.0, action_mode='continuous', setpoint=None):
                 
        kwargs = {'timestep': timestep,
                  'beam_length': beam_length,
                  'max_angle':max_angle, 
                  'action_mode':action_mode, 
                  'init_velocity': init_velocity}

        super().__init__(**kwargs)

        # random setpoint for None values
        if setpoint is None:
            self.setpoint = np.random.random_sample()*beam_length - beam_length/2
            self.random_setpoint = True
        else:
            if abs(setpoint) > beam_length/2:
                raise ValueError('Setpoint outside of beam.')
            self.setpoint = setpoint
            self.random_setpoint = False
                                                            # [angle, position, velocity, setpoint]
        self.observation_space = spaces.Box(low=np.array([-max_angle, -np.inf, -np.inf, -beam_length/2]), 
                                            high=np.array([max_angle, np.inf, np.inf, beam_length/2]))

    def step(self, action):
        """
        Update environment for one action

        Parameters
        ----------
        action [continuous] : set angle, float (rad)
        action [discrete] : increase/descrease angle, int [0, 1]
        """
        self.bb.update(self._action_conversion(action))
        obs = np.array([self.bb.theta, self.bb.x, self.bb.v, self.setpoint])

        # reward squared proximity to setpoint 
        reward = (1.0 - (self.setpoint - self.bb.x)/self.bb.L)**2
        
        return obs, reward, not self.bb.on_beam, {}

    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        super().reset()
        
        if self.random_setpoint is None:
            self.setpoint = np.random.random_sample()*self.beam_length \
                            - self.beam_length/2

        return np.array([self.bb.theta, self.bb.x, self.bb.v, self.setpoint])

class VisualBallBeamSetpointEnv(VisualBallBeamBaseEnv):
    """ VisualBallBeamSetpointEnv

    Setpoint environment with simulation plot as state

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    action_mode : action space, str ['continuous', 'discrete']

    setpoint : target position of ball, float (units)
    """
    
    def __init__(self, timestep=0.1, beam_length=1.0, max_angle=0.2, 
                 init_velocity=0.0, action_mode='continuous', setpoint=None):

        kwargs = {'timestep': timestep,
                  'beam_length': beam_length,
                  'max_angle':max_angle, 
                  'action_mode':action_mode, 
                  'init_velocity': init_velocity}

        super().__init__(**kwargs)

        # random setpoint for None values
        if setpoint is None:
            self.setpoint = np.random.random_sample()*beam_length - beam_length/2
            self.random_setpoint = True
        else:
            if abs(setpoint) > beam_length/2:
                raise ValueError('Setpoint outside of beam.')
            self.setpoint = setpoint
            self.random_setpoint = False

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

        # reward squared proximity to setpoint 
        reward = (1.0 - (self.setpoint - self.bb.x)/self.bb.L)**2
        
        return obs, reward, not self.bb.on_beam, {}

    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        
        if self.random_setpoint is None:
            self.setpoint = np.random.random_sample()*self.beam_length \
                            - self.beam_length/2

        return super().reset()



