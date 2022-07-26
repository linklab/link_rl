import sys

from gym.envs.registration import register

default_max_episode_steps = 10_000_000

register(
    id="InHandManipulation-v0",
    entry_point="environments.InHandManipulation.InHandManipulation:InHandManipulation",
    max_episode_steps=default_max_episode_steps,
)
