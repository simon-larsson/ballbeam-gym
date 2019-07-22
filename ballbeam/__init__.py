from gym.envs.registration import register

register(
    id='BallBeamBalance-v0',
    entry_point='ballbeam.envs:BallBeamBalanceEnv',
)

register(
    id='BallBeamSetpoint-v0',
    entry_point='ballbeam.envs:BallBeamSetpointEnv',
)
