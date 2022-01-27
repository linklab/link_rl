from gym.spaces import Box, Discrete

from g_utils.types import AgentType


def get_agent(observation_space, action_space, config=None):
    assert isinstance(observation_space, Box)

    if config.AGENT_TYPE == AgentType.DQN:
        assert isinstance(action_space, Discrete)
        from d_agents.off_policy.dqn.agent_dqn import AgentDqn
        agent = AgentDqn(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.DOUBLE_DQN:
        from d_agents.off_policy.dqn.agent_double_dqn import AgentDoubleDqn
        agent = AgentDoubleDqn(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.DUELING_DQN:
        from d_agents.off_policy.dqn.agent_dueling_dqn import AgentDuelingDqn
        agent = AgentDuelingDqn(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.DOUBLE_DUELING_DQN:
        from d_agents.off_policy.dqn.agent_double_dueling_dqn import AgentDoubleDuelingDqn
        agent = AgentDoubleDuelingDqn(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.REINFORCE:
        assert config.N_ACTORS * config.N_VECTORIZED_ENVS == 1, "TOTAL NUMBERS OF ENVS should be one"
        from d_agents.on_policy.reinforce.agent_reinforce import AgentReinforce
        agent = AgentReinforce(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.A2C:
        from d_agents.on_policy.a2c.agent_a2c import AgentA2c
        agent = AgentA2c(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.PPO:
        from d_agents.on_policy.ppo.agent_ppo import AgentPpo
        agent = AgentPpo(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.PPO_TRAJECTORY:
        from d_agents.on_policy.ppo.agent_ppo_trajectory import AgentPpoTrajectory
        assert hasattr(config, "PPO_TRAJECTORY_SIZE")
        assert config.PPO_TRAJECTORY_SIZE % config.BATCH_SIZE == 0, "{0} {1}".format(
            config.PPO_TRAJECTORY_SIZE, config.BATCH_SIZE
        )
        agent = AgentPpoTrajectory(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.DDPG:
        from d_agents.off_policy.ddpg.agent_ddpg import AgentDdpg
        agent = AgentDdpg(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.TD3:
        from d_agents.off_policy.td3.agent_td3 import AgentTd3
        agent = AgentTd3(
            observation_space=observation_space, action_space=action_space, config=config
        )
    elif config.AGENT_TYPE == AgentType.SAC:
        from d_agents.off_policy.sac.agent_sac import AgentSac
        agent = AgentSac(
            observation_space=observation_space, action_space=action_space, config=config
        )
    else:
        raise ValueError()

    return agent
