import gym
from ballbeam.envs.ballbeam_setpoint_env import BallBeamSetpointEnv

MAX_ANGLE = 0.2

env = BallBeamSetpointEnv(time_step=0.05, setpoint=0.5, max_angle=MAX_ANGLE)

Kp = 2.0
Kd = 1.0

for i in range(1000):
    theta = Kp*(env.bb.x - env.setpoint) + Kd*(env.bb.v)
    obs, reward, done, info = env.step(theta)
    env.render()

    if done:
        env.reset()

env.close()