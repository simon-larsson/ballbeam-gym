[![PyPI version](https://badge.fury.io/py/ballbeam-gym.svg)](https://pypi.python.org/pypi/ballbeam-gym/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/simon-larsson/ballbeam-gym/blob/master/LICENSE)

# Ball &amp; Beam Gym
Ball & beam simulation as OpenAI gym environments.

---

## Installation

Run command:

    pip install ballbeam-gym

or clone the repository and run the following inside the folder:

    pip install -e .

---

## System Dynamics
Simulated as a frictionless first order system that takes the beam angle as input. The equation that describe the system is as follows:

    dx/dt = v(t)
    dv/dt = -m*g*sin(theta(t))/((I + 1)*m)

[![visualization](ballbeam.png)](https://github.com/simon-larsson/ballbeam-gym)

---

## Environments
- [BallBeamBalanceEnv](https://github.com/simon-larsson/ballbeam-gym#ballbeambalanceenv) - Objective is to not drop the ball from the beam using key state variables as observation space.
- [VisualBallBeamBalanceEnv](https://github.com/simon-larsson/ballbeam-gym#visualballbeambalanceenv) - Same as above but only using simulation plot as observation space.
- [BallBeamSetpointEnv](https://github.com/simon-larsson/ballbeam-gym#ballbeamsetpointenv) - Objective is to keep the ball as close to a set position on the beam as possible using key state variables as observation space.
- [VisualBallBeamSetpointEnv](https://github.com/simon-larsson/ballbeam-gym#visualballbeamsetpointenv) - Same as above but only using simulation plot as observation space.

#### Alias
- `BallBeamBalance-v0`
- `VisualBallBeamBalance-v0`
- `BallBeamSetpoint-v0`
- `VisualBallBeamSetpoint-v0`

---

## API

The environments use the same API and inherits from OpenAI gyms.
- `step(action)` - Simulate one timestep.
- `reset()` - Reset environment to start conditions.
- `render(mode='human')` - Visualize one timestep.
- `seed(seed)` - Make environment deterministic.

---

### BallBeamBalanceEnv

Ball is given a random or set initial velocity and it is the agents job to stabilize the ball on the beam using a set of key state variables.

**Parameters**
- `timestep` - Length of a timestep.
- `beam_length` - Length of beam.
- `max_angle` - Max abs(angle) of beam.
- `init_velocity` - Initial speed of ball (`None` for random).
- `action_mode` - Continuous or discrete action space.

**Observation Space** 
- Beam angle
- Ball position on beam.
- Ball velocity

**Action Space**

Continuous:
- Beam angle

Discrete:
- Increase angle
- Descrease angle

**Rewards**

A reward of +1 is given for each timestep ball stays on beam.

**Reset**

Resets when ball falls of beam.

---

### VisualBallBeamBalanceEnv

Ball is given a random or set initial velocity and it is the agents job to stabilize the ball on the beam using a image data from the simulation plot.

**Parameters**
- `timestep` - Length of a timestep.
- `beam_length` - Length of beam.
- `max_angle` - Max abs(angle) of beam.
- `init_velocity` - Initial speed of ball (`None` for random).
- `action_mode` - Continuous or discrete action space.

**Observation Space** 
- RGB image [200x250x3]

**Action Space**

Continuous:
- Beam angle

Discrete:
- Increase angle
- Descrease angle

**Rewards**

A reward of +1 is given for each timestep ball stays on beam.

**Reset**

Resets when ball falls of beam.

---

### BallBeamSetpointEnv

The agent's job is to keep the ball's position as close as possible to a setpoint using a set of key state variables.

**Parameters**
- `timestep` - Length of a timestep.
- `beam_length` - Length of beam.
- `max_angle` - Max abs(angle) of beam.
- `init_velocity` - Initial speed of ball (`None` for random).
- `action_mode` - Continuous or discrete action space.
- `setpoint` - Target position of ball (`None` for random).

**Observation Space** 
- Beam angle
- Ball position
- Ball velocity
- Setpoint position

**Action Space**

Continuous:
- Beam angle

Discrete:
- Increase angle
- Descrease angle

**Rewards**

At each timestep the agent is rewarded with the squared proximity between the ball and the setpoint: 

`reward = (1 - (setpoint - ball_position)/beam_length)^2`.

**Reset**

Resets when ball falls of beam.

---

### VisualBallBeamSetpointEnv

The agent's job is to keep the ball's position as close as possible to a setpoint using a image data from the simulation plot.

**Parameters**
- `timestep` - Length of a timestep.
- `beam_length` - Length of beam.
- `max_angle` - Max abs(angle) of beam.
- `init_velocity` - Initial speed of ball (`None` for random).
- `action_mode` - Continuous or discrete action space.
- `setpoint` - Target position of ball (`None` for random).

**Observation Space** 
- RGB image [200x250x3]

**Action Space**

Continuous:
- Beam angle

Discrete:
- Increase angle
- Descrease angle

**Rewards**

At each timestep the agent is rewarded with the squared proximity between the ball and the setpoint: 

`reward = (1 - (setpoint - ball_position)/beam_length)^2`.

**Reset**

Resets when ball falls of beam.

---

## Example: PID Controller
```python
import gym
import ballbeam_gym

# pass env arguments as kwargs
kwargs = {'timestep': 0.05, 
          'setpoint': 0.4,
          'beam_length': 1.0,
          'max_angle': 0.2,
          'init_velocity': 0.0,
          'action_mode': 'continuous'}

# create env
env = gym.make('BallBeamSetpoint-v0', **kwargs)

# constants for PID calculation
Kp = 2.0
Kd = 1.0

# simulate 1000 steps
for i in range(1000):   
    # control theta with a PID controller
    env.render()
    theta = Kp*(env.bb.x - env.setpoint) + Kd*(env.bb.v)
    obs, reward, done, info = env.step(theta)

    if done:
        env.reset()

```

## Example: Reinforcement Learning
```python
import gym
import ballbeam_gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# pass env arguments as kwargs
kwargs = {'timestep': 0.05, 
          'setpoint': 0.4,
          'beam_length': 1.0,
          'max_angle': 0.2,
          'init_velocity': 0.0,
          'action_mode': 'discrete'}

# create env
env = gym.make('BallBeamSetpoint-v0', **kwargs)

# train a mlp policy agent
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
env.render()

# test agent on 1000 steps
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
