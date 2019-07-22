import gym
from ballbeam.envs.ballbeam_setpoint_env import BallBeamSetpointEnv

TIME_STEP = 0.05
SETPOINT = 0.5
MAX_ANGLE = 0.2

env = BallBeamSetpointEnv(time_step=TIME_STEP, 
                          setpoint=SETPOINT,
                          max_angle=MAX_ANGLE)

Kp = 2.0
Ki = 0.0
Kd = 1.0

class PID():

    def __init__(self, setpoint, Kp=2.0, Ki = 0.0, Kd=1.0):
        self.setpoint = setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.err = 0.0
        self.I_err = 0.0

    def update(self, value, dt):
        error = value - self.setpoint
        delta_error = error - self.err
        self.I_err += error*dt
        self.err = error

        return self.Kp*error + self.Ki*self.I_err + (delta_error*self.Kd) / dt

    def reset(self):
        self.err = 0.0
        self.I_err = 0.0

pid = PID(SETPOINT, Kp, Ki, Kd)

for i in range(1000):
    theta = pid.update(env.bb.x, 0.05)
    obs, reward, done, info = env.step(theta)
    env.render()

    if done:
        pid.reset()
        env.reset()

env.close()





