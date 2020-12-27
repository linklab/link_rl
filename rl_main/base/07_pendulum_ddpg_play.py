#!/usr/bin/env python3
import gym
import torch
import os

from config.names import DeepLearningModelName, PROJECT_HOME

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")

print(torch.__version__)

from common.fast_rl.rl_agent import float32_preprocessor
from common.fast_rl import actions, value_based_model, rl_agent, policy_based_model, experience
import numpy as np

from config.parameters import PARAMETERS as params

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


STEP_LENGTH = 2

def play_main():
    env = gym.make(params.ENVIRONMENT_ID.value)

    print(env.action_space.low[0], env.action_space.high[0])
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    if params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_ACTOR_CRITIC_MLP:
        actor_net = policy_based_model.DDPGActor(
            obs_size=3,
            hidden_size_1=512, hidden_size_2=256,
            n_actions=1,
            scale=2.0
        ).to(device)
    elif params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_ACTOR_CRITIC_GRU:
        actor_net = policy_based_model.DDPGGruActor(
            obs_size=3,
            hidden_size_1=128, hidden_size_2=64,
            n_actions=1,
            bidirectional=False,
            scale=2.0
        ).to(device)
    elif params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_ACTOR_CRITIC_GRU_ATTENTION:
        actor_net = policy_based_model.DDPGGruAttentionActor(
            obs_size=3,
            hidden_size=128,
            n_actions=1,
            bidirectional=False,
            scale=2.0
        ).to(device)
    else:
        raise ValueError()

    print(actor_net)

    rl_agent.load_model(MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, actor_net.__name__, actor_net)

    action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=0.0, ou_enabled=False, scale_factor=1.0)

    agent = rl_agent.AgentDDPG(
        actor_net, n_actions=1, action_selector=action_selector,
        action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
    )

    if params.DEEP_LEARNING_MODEL in [DeepLearningModelName.DDPG_ACTOR_CRITIC_GRU, DeepLearningModelName.DDPG_ACTOR_CRITIC_GRU_ATTENTION]:
        step_length = params.RNN_STEP_LENGTH
    else:
        step_length = -1

    experience_source = experience.ExperienceSourceSingleEnvFirstLast(
        env, agent, gamma=params.GAMMA, steps_count=params.N_STEP,
        step_length=step_length, render=True
    )

    exp_source_iter = iter(experience_source)

    done = False

    while not done:
        exp = next(exp_source_iter)
        done = exp.done


if __name__ == "__main__":
    play_main()