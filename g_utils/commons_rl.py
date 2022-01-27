from gym.spaces import Box, Discrete
from torch import nn
import torch.nn.functional as F
from g_utils.types import ModelType, LayerActivationType, LossFunctionType, AgentType


def set_config(config):
    if config.MODEL_TYPE in (
            ModelType.TINY_LINEAR, ModelType.SMALL_LINEAR, ModelType.SMALL_LINEAR_2,
            ModelType.MEDIUM_LINEAR, ModelType.LARGE_LINEAR
    ):
        from a_configuration.a_base_config.c_models.linear_models import ConfigLinearModel
        config.MODEL_PARAMETER = ConfigLinearModel(config.MODEL_TYPE)
    elif config.MODEL_TYPE in (
            ModelType.SMALL_CONVOLUTIONAL, ModelType.MEDIUM_CONVOLUTIONAL, ModelType.LARGE_CONVOLUTIONAL
    ):
        from a_configuration.a_base_config.c_models.convolutional_models import ConfigConvolutionalModel
        config.MODEL_PARAMETER = ConfigConvolutionalModel(config.MODEL_TYPE)
    elif config.MODEL_TYPE in (
            ModelType.SMALL_RECURRENT, ModelType.MEDIUM_RECURRENT, ModelType.LARGE_RECURRENT
    ):
        from a_configuration.a_base_config.c_models.recurrent_linear_models import ConfigRecurrentLinearModel
        config.MODEL_PARAMETER = ConfigRecurrentLinearModel(config.MODEL_TYPE)
    elif config.MODEL_TYPE in (
            ModelType.SMALL_RECURRENT_CONVOLUTIONAL, ModelType.MEDIUM_RECURRENT_CONVOLUTIONAL,
            ModelType.LARGE_RECURRENT_CONVOLUTIONAL
    ):
        from a_configuration.a_base_config.c_models.recurrent_convolutional_models import ConfigRecurrentConvolutionalModel
        config.MODEL_PARAMETER = ConfigRecurrentConvolutionalModel(config.MODEL_TYPE)
    else:
        raise ValueError()

    if config.LAYER_ACTIVATION_TYPE == LayerActivationType.LEAKY_RELU:
        config.LAYER_ACTIVATION = nn.LeakyReLU
    elif config.LAYER_ACTIVATION_TYPE == LayerActivationType.ELU:
        config.LAYER_ACTIVATION = nn.ELU
    else:
        raise ValueError()

    if config.LOSS_FUNCTION_TYPE == LossFunctionType.MSE_LOSS:
        config.LOSS_FUNCTION = F.mse_loss
    elif config.LOSS_FUNCTION_TYPE == LossFunctionType.HUBER_LOSS:
        config.LOSS_FUNCTION = F.huber_loss
    else:
        raise ValueError()


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
        assert config.PPO_TRAJECTORY_SIZE % config.BATCH_SIZE == 0, "{0} {1}".format(
            config.PPO_TRAJECTORY_SIZE, config.BATCH_SIZE
        )
        agent = AgentPpo(
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
