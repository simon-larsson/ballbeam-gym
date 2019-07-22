import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from ballbeam.envs.ballbeam_balance_env import BallBeamBalanceEnv
from ballbeam.envs.ballbeam_setpoint_env import BallBeamSetpointEnv

#env = BallBeamBalanceEnv(time_step=0.05)
env = BallBeamSetpointEnv(time_step=0.05, setpoint=0.4)
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=20000)

obs = env.reset()
env.render()

for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()

env.close()