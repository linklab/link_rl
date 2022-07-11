import gym
from gym.envs.registration import register


def make(
        domain_name,
        task_name,
        seed=1,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=4,
        episode_length=1000,
        environment_kwargs=None,
        time_limit=None,
        frame_stack=1,
        grayscale=True
):
    env_id = 'dmc_%s_%s_%s-v1' % (domain_name, task_name, seed)

    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # shorten episode length
    # frame_skip = 1 --> max_episode_steps: 1_000 // 1 = 1_000
    # frame_skip = 2 --> max_episode_steps: 1_001 // 2 = 500
    # frame_skip = 3 --> max_episode_steps: 1_002 // 3 = 334
    # frame_skip = 4 --> max_episode_steps: 1_003 // 4 = 250
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    #print("**** max_episode_steps:", max_episode_steps)

    if not env_id in gym.envs.registry.env_specs:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        register(
            id=env_id,
            entry_point='link_rl.b_environments.dm_control.dmc_wrapper:DMCWrapper',
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                frame_stack=frame_stack,
                grayscale=grayscale
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)