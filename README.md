[![PyPI version](https://badge.fury.io/py/ballbeam-gym.svg)](https://pypi.python.org/pypi/ballbeam-gym/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/simon-larsson/ballbeam-gym/blob/master/LICENSE)

# Ball &amp; Beam Gym
Ball & beam simulation as OpenAI gym environments.

## Installation
Clone the repository and run the following inside the folder:

    pip install -e .


## System Dynamics
Simulated as a first order system that takes the beam angle as input. The equation that describe the system is as follows:

    dx/dt = v(t)
    dv/dt = -m*g*sin(theta(t))/((I + 1)*m)

[![visualization](ballbeam.png)](https://github.com/simon-larsson/ballbeam-gym)

## Environments
- **BallBeamBalanceEnv** - Objective is to not drop the ball from the beam.
- **BallBeamSetpointEnv** - Objective is to keep the ball as close to a set position on the beam as possible.

## Example
```python
import gym

# pass env arguments as kwargs
kwargs = {'time_step': 0.05, 
          'setpoint': 0.4,
          'beam_length': 1.0,
          'max_angle': 0.2}

# create env
env = gym.make('BallBeamSetpoint-v0', **kwargs)

# constants for PID calculation
Kp = 2.0
Kd = 1.0

# simulate 1000 steps
for i in range(1000):   
    # control theta with a PID controller
    theta = Kp*(env.bb.x - env.setpoint) + Kd*(env.bb.v)
    obs, reward, done, info = env.step(theta)
    env.render()

```
