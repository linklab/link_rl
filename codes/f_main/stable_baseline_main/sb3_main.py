# https://github.com/DLR-RM/stable-baselines3
# pip install stable-baselines3
import gym
import os
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import EvalCallback

from codes.b_environments.rotary_inverted_pendulum.rip import RotaryInvertedPendulumEnv
from codes.a_config.parameters import PARAMETERS as params
from codes.e_utils.names import EnvironmentName
from codes.f_main.stable_baseline_main.sb_callback import Callback
from stable_baselines3.common.env_util import make_vec_env

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = RotaryInvertedPendulumEnv(
    action_min=params.ACTION_SCALE * -1.0,
    action_max=params.ACTION_SCALE,
    env_reset=params.ENV_RESET,
    pendulum_type=EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0,
    params=params
)
env.start()

# env = gym.make('LunarLander-v2')

model = TD3("MlpPolicy", env, device='cuda', verbose=0)

callback = Callback(model)
eval_callback = EvalCallback(eval_env=env)
model.learn(total_timesteps=10000000, callback=[callback, eval_callback])

obs = env.reset()
for i in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()