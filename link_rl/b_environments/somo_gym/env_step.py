import os
import sys

import wandb
import yaml
from pathlib import Path

import gym

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from link_rl.b_environments.somo_gym.environments.utils.import_handler import import_environment


def somogym_step_tester(env_name, render=False, debug=False):
    from link_rl.a_configuration.b_single_config.somo_gym.config_somo_gym_in_hand_manipulation import \
        ConfigSomoGymInHandManipulationSac

    config = ConfigSomoGymInHandManipulationSac()

    os.environ['WANDB_START_METHOD'] = 'thread'
    wandb_obj = wandb.init()
    print("!!!!!!!!!!!!!")
    # wandb_obj = wandb.init(
    #     project="somo_gym_test",
    #     # config={
    #     #     key: getattr(config, key) for key in dir(config) if not key.startswith("__")
    #     # }
    # )

    run_config_file = (
        Path(os.path.dirname(__file__))
        / "environments"
        / env_name
        / "benchmark_run_config.yaml"
    )

    with open(run_config_file, "r") as config_file:
        run_config = yaml.safe_load(config_file)

    # prepare env
    import_environment(env_name)
    env = gym.make(
        run_config["env_id"],
        run_config=run_config,
        run_ID=f"{env_name}-step_test",
        render=render,
        debug=debug,
    )

    print(env.observation_space)
    print(env.action_space)

    for ep in range(1, 3):
        run_config["seed"] = 10110
        env.seed(run_config["seed"])
        observation = env.reset()

        done = False
        step = 1

        # run env for total_env_steps steps
        while not done:
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)  # take a random action
            print(
                "[EP: {0}, STEP: {1}] Observation: {2}, Action: {3}, next_observation: {4}, Reward: {5:.5f}, Done: {6}".format(
                    ep, step, observation.shape, action.shape, next_observation.shape, reward, done
                ))
            observation = next_observation
            step += 1

    # make sure seeding works correctly for this env
    # seed once, reset, and take a step
    env.seed(run_config["seed"])
    env.reset()
    action_a = env.action_space.sample()
    step_result_a = env.step(action_a)  # take a random action

    # seed and reset again and take another step
    env.seed(run_config["seed"])
    env.reset()
    action_b = env.action_space.sample()
    step_result_b = env.step(action_b)  # take a random action

    # compare results
    assert (
        step_result_a[0] == step_result_b[0]
    ).all(), f"seeding does not work correctly for env {env_name}: observations are inconsistent"
    assert (
        step_result_a[1] == step_result_b[1]
    ), f"seeding does not work correctly for env {env_name}: rewards are inconsistent"
    assert (
        step_result_a[2] == step_result_b[2]
    ), f"seeding does not work correctly for env {env_name}: done flags are inconsistent"
    assert (
        step_result_a[3] == step_result_b[3]
    ), f"seeding does not work correctly for env {env_name}: info entries are inconsistent"

    # finally, close the env
    env.close()


if __name__ == "__main__":
    somogym_step_tester("InHandManipulation", render=False, debug=False)

