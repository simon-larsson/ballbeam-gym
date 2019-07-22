import gym
from ballbeam.envs.ballbeam_setpoint_env import BallBeamSetpointEnv

TIME_STEP = 0.05
SETPOINT = None
MAX_ANGLE = 0.2
BEAM_LENGTH = 1.0

env = BallBeamSetpointEnv(time_step=TIME_STEP, 
                          setpoint=SETPOINT,
                          beam_length=BEAM_LENGTH,
                          max_angle=MAX_ANGLE)

Kp = 2.0
Kd = 1.0

for i in range(1000):
    # PID-calculation (I-part skipped since it only hurts here)  
    theta = Kp*(env.bb.x - env.setpoint) + Kd*(env.bb.v)
    obs, reward, done, info = env.step(theta)
    env.render()
    if done:
        env.reset()

env.close()





