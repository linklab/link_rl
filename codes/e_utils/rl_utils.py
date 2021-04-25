import random

from gym import Env
from gym.spaces import Box, Discrete
from gym.vector import VectorEnv
from numpy import random

import gym
import torch
from torch import optim
import os, sys

from codes.a_config.f_trade_parameters.parameters_trade_dqn import PARAMETERS_GENERAL_TRADE_DQN
from codes.a_config.parameters import PARAMETERS as params

from codes.b_environments.custom_sync_vector_env import CustomSyncVectorEnv
from codes.b_environments.trade.trade_data import get_data
from codes.c_models.continuous_action.soft_actor_critic_model import SoftActorCriticModel
from codes.d_agents.black_box.cma_es.cma_es_agent import AgentEMAES

from codes.d_agents.black_box.ga.ga_agent import AgentGA
from codes.d_agents.black_box.ga.multi_ga_agent import AgentMultiGA
from codes.d_agents.off_policy.td3.td3_agent import AgentTD3
from codes.d_agents.off_policy.sac.continuous_sac_agent import AgentSAC
from codes.d_agents.on_policy.ppo.discrete_ppo_agent import AgentDiscretePPO
from codes.e_utils.reward_changer import PseudoCountRewardWrapper

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.d_agents.on_policy.a2c.continuous_a2c_agent import AgentContinuousA2C
from codes.d_agents.on_policy.ppo.continuous_ppo_agent import AgentContinuousPPO
from codes.d_agents.on_policy.a2c.discrete_a2c_agent import AgentDiscreteA2C
from codes.d_agents.off_policy.dqn.dqn_agent import AgentDQN

from codes.c_models.continuous_action.deterministic_continuous_actor_critic_model import DeterministicContinuousActorCriticModel
from codes.c_models.continuous_action.stochastic_continuous_actor_critic_model import StochasticContinuousActorCriticModel
from codes.c_models.discrete_action.discrete_actor_critic_model import DiscreteActorCriticModel
from codes.c_models.discrete_action.dqn_model import DuelingDQNModel

from codes.d_agents.off_policy.ddpg.ddpg_agent import AgentDDPG

from codes.e_utils.common_utils import make_atari_env
from codes.e_utils.names import EnvironmentName, DeepLearningModelName, RLAlgorithmName, OptimizerName, AgentMode


MODEL_ZOO_SAVE_DIR = os.path.join(PROJECT_HOME, "codes", "g_play", "model_zoo")

if isinstance(params, PARAMETERS_GENERAL_TRADE_DQN):
    MODEL_SAVE_FILE_PREFIX = "_".join([params.ENVIRONMENT_ID.value, params.COIN_NAME, params.TIME_UNIT])
else:
    MODEL_SAVE_FILE_PREFIX = params.ENVIRONMENT_ID.value


def get_environment(params):
    def make_environment(params):
        def _make():
            env = get_single_environment(params=params, mode=AgentMode.TRAIN)
            if params.COUNT_BASED_EXPLORATION:
                assert len(env.observation_space.shape) == 1, "env.observation_space.shape should be one"

                if not params.COUNT_BASED_FILTER:
                    params.COUNT_BASED_FILTER = [1] * env.observation_space.shape[0]
                assert len(params.COUNT_BASED_FILTER) == env.observation_space.shape[0], \
                    "Current params.COUNT_BASED_FILTER: {0} and params.env.observation_space.shape: {1}".format(
                        params.COUNT_BASED_FILTER, env.observation_space.shape
                    )

                env = PseudoCountRewardWrapper(env=env, params=params)
            return env

        return _make
    env_fns = [make_environment(params=params) for _ in range(params.NUM_ENVIRONMENTS)]

    # 매 타임 스텝마다 모든 env들로 부터 transition을 가져옴.
    # 각 env에 대한 통신은 parallel 하지 않음.
    env = CustomSyncVectorEnv(env_fns)
    assert env.num_envs == params.NUM_ENVIRONMENTS
    return env


def get_single_environment(params=None, mode=AgentMode.TRAIN):
    if params.ENVIRONMENT_ID == EnvironmentName.REAL_DEVICE_RIP:
        from codes.b_environments.rotary_inverted_pendulum.rip import RotaryInvertedPendulumEnv
        env = RotaryInvertedPendulumEnv(
            action_min=params.ACTION_SCALE * -1.0,
            action_max=params.ACTION_SCALE,
            env_reset=params.ENV_RESET,
            pendulum_type=EnvironmentName.REAL_DEVICE_RIP,
            params=params
        )
    elif params.ENVIRONMENT_ID == EnvironmentName.REAL_DEVICE_DOUBLE_RIP:
        from codes.b_environments.rotary_inverted_pendulum.rip import RotaryInvertedPendulumEnv
        env = RotaryInvertedPendulumEnv(
            action_min=params.ACTION_SCALE * -1.0,
            action_max=params.ACTION_SCALE,
            env_reset=params.ENV_RESET,
            pendulum_type=EnvironmentName.REAL_DEVICE_DOUBLE_RIP,
            params=params
        )

    elif params.ENVIRONMENT_ID == EnvironmentName.QUANSER_SERVO_2:
        from codes.b_environments.quanser_rotary_inverted_pendulum.quanser_rip import EnvironmentQuanserRIP
        env = EnvironmentQuanserRIP(
            action_min=params.ACTION_SCALE * -1.0,
            action_max=params.ACTION_SCALE
        )
    elif params.ENVIRONMENT_ID in [
        EnvironmentName.CARTPOLE_V0, EnvironmentName.CARTPOLE_V1,
        EnvironmentName.MOUNTAINCAR_V0, EnvironmentName.MOUNTAINCARCONTINUOUS_V0,
        EnvironmentName.ACROBOT_V1, EnvironmentName.BLACKJACK_V0,
        EnvironmentName.INVERTED_DOUBLE_PENDULUM_V2, EnvironmentName.HOPPER_V2, EnvironmentName.SWIMMER_V2,
        EnvironmentName.REACHER_V2, EnvironmentName.HUMANOID_V2, EnvironmentName.HUMANOID_STAND_UP_V2,
        EnvironmentName.INVERTED_PENDULUM_V2, EnvironmentName.WALKER_2D_V2, EnvironmentName.LUNAR_LANDER_V2,
        EnvironmentName.LUNAR_LANDER_CONTINUOUS_V2
    ]:
        env = gym.make(params.ENVIRONMENT_ID.value)
    elif params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_V0]:
        env = gym.make(params.ENVIRONMENT_ID.value)
        #env = RewardChanger(env, lambda r: (r + 8.0) / 8.0, lambda r: 8.0 * r - 8.0)
    elif params.ENVIRONMENT_ID == EnvironmentName.FROZENLAKE_V0:
        env = gym.make(EnvironmentName.FROZENLAKE_V0.value, is_slippery=False)
    elif params.ENVIRONMENT_ID == EnvironmentName.CHASER_V1_MAC or params.ENVIRONMENT_ID == EnvironmentName.CHASER_V1_WINDOWS:
        from codes.b_environments.unity.chaser_unity import Chaser_v1
        env = Chaser_v1(params.MY_PLATFORM)
    elif params.ENVIRONMENT_ID in [
        EnvironmentName.BREAKOUT_DETERMINISTIC_V4,
        EnvironmentName.BREAKOUT_NO_FRAME_SKIP_V4,
        EnvironmentName.PONG_NO_FRAME_SKIP_V4,
        EnvironmentName.ENDURO_NO_FRAME_SKIP_V4,
        EnvironmentName.SEAQUEST_NO_FRAME_SKIP_V4,
        EnvironmentName.FREEWAY_NO_FRAME_SKIP_V4,
        EnvironmentName.KUNGFU_MASTER_FRAME_SKIP_V4
    ]:
        env = make_atari_env(params.ENVIRONMENT_ID.value)
    elif params.ENVIRONMENT_ID in [EnvironmentName.DRONE_RACING_MAC, EnvironmentName.DRONE_RACING_WINDOWS]:
        from codes.b_environments.unity.drone_racing import Drone_Racing
        env = Drone_Racing(params.MY_PLATFORM)
    elif params.ENVIRONMENT_ID in [
        EnvironmentName.PYBULLET_ANT_V0, EnvironmentName.PYBULLET_HALF_CHEETAH_V0,
        EnvironmentName.PYBULLET_INVERTED_DOUBLE_PENDULUM_V0
    ]:
        env = gym.make(params.ENVIRONMENT_ID.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.PENDULUM_MATLAB_V0:
        from codes.b_environments.rotary_inverted_pendulum.rip import RotaryInvertedPendulumEnv
        env = RotaryInvertedPendulumEnv(
            action_min=params.ACTION_SCALE * -1.0,
            action_max=params.ACTION_SCALE,
            env_reset=params.ENV_RESET,
            pendulum_type=EnvironmentName.PENDULUM_MATLAB_V0,
            params=params
        )
        env.start()
    elif params.ENVIRONMENT_ID == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
        from codes.b_environments.rotary_inverted_pendulum.rip import RotaryInvertedPendulumEnv
        env = RotaryInvertedPendulumEnv(
            action_min=params.ACTION_SCALE * -1.0,
            action_max=params.ACTION_SCALE,
            env_reset=params.ENV_RESET,
            pendulum_type=EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0,
            params=params
        )
        env.start()
    elif params.ENVIRONMENT_ID == EnvironmentName.TRADE_V0:
        from codes.b_environments.trade.trade_env import UpbitEnvironment
        from codes.b_environments.trade.trade_constant import TradeEnvironmentType

        assert hasattr(params, "COIN_NAME")
        assert hasattr(params, "TIME_UNIT")

        train_data_info, evaluate_data_info = get_data(coin_name=params.COIN_NAME, time_unit=params.TIME_UNIT)

        print(train_data_info["first_datetime_krw"], train_data_info["last_datetime_krw"])
        print(evaluate_data_info["first_datetime_krw"], evaluate_data_info["last_datetime_krw"])

        if mode == AgentMode.TRAIN:
            env = UpbitEnvironment(
                coin_name=params.COIN_NAME, time_unit=params.TIME_UNIT,
                data_info=train_data_info, params=params, environment_type=TradeEnvironmentType.TRAIN
            )
        elif mode == AgentMode.TEST:
            env = UpbitEnvironment(
                coin_name=params.COIN_NAME, time_unit=params.TIME_UNIT,
                data_info=evaluate_data_info, params=params, environment_type=TradeEnvironmentType.TEST_RANDOM
            )
        elif mode == AgentMode.PLAY:
            env = UpbitEnvironment(
                coin_name=params.COIN_NAME, time_unit=params.TIME_UNIT,
                data_info=None, params=params, environment_type=TradeEnvironmentType.LIVE
            )
        else:
            raise ValueError()
    else:
        env = None

    if hasattr(env, "seed") and hasattr(params, "SEED") and params.SEED is not None:
        env.seed(params.SEED)
    else:
        env.seed(random.randint(sys.maxsize))

    return env


def get_environment_input_output_info(env):
    if isinstance(env, VectorEnv):
        input_shape = env.single_observation_space.shape
        action_shape = env.single_action_space.shape
        if isinstance(env.single_action_space, Discrete):
            num_outputs = env.single_action_space.n
            action_min, action_max = None, None
        elif isinstance(env.single_action_space, Box):
            num_outputs = env.single_action_space.shape[0]
            action_min = env.single_action_space.low[0]
            action_max = env.single_action_space.high[0]
        else:
            num_outputs, action_shape, action_min, action_max = None, None, None, None
    elif isinstance(env, Env):
        input_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        if isinstance(env.action_space, Discrete):
            num_outputs = env.action_space.n
            action_min, action_max = None, None
        elif isinstance(env.action_space, Box):
            num_outputs = env.action_space.shape[0]
            action_min = env.action_space.low[0]
            action_max = env.action_space.high[0]
        else:
            num_outputs, action_shape, action_min, action_max = None, None, None, None
    else:
        raise ValueError()

    if action_min and action_max:
        print(f"input_shape: {input_shape}, action_shape: {action_shape}, "
              f"num_outputs: {num_outputs}, action_min: {action_min}, action_max: {action_max}")
    else:
        print(f"input_shape: {input_shape}, action_shape: {action_shape}, num_outputs: {num_outputs}")

    return input_shape, action_shape, num_outputs, action_min, action_max


def get_rl_model(worker_id, input_shape=None, num_outputs=None, params=None, device=None):
    if params.DEEP_LEARNING_MODEL == DeepLearningModelName.SOFT_ACTOR_CRITIC_MLP:
        model = SoftActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP:
        model = StochasticContinuousActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL in [
        DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP,
        DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_CNN
    ]:
        model = DiscreteActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP:
        model = DeterministicContinuousActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL in [
        DeepLearningModelName.DUELING_DQN_MLP,
        DeepLearningModelName.DUELING_DQN_CNN,
        DeepLearningModelName.DUELING_DQN_SMALL_CNN
    ]:
        model = DuelingDQNModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.NO_MODEL:
        model = None
    else:
        model = None
    return model


def get_rl_agent(input_shape, action_shape, num_outputs, action_min, action_max, worker_id, params, device="cpu"):
    if params.RL_ALGORITHM == RLAlgorithmName.DDPG_V0:
        agent = AgentDDPG(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            action_min=action_min, action_max=action_max, params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.TD3_V0:
        agent = AgentTD3(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            action_min=action_min, action_max=action_max, params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.DQN_V0:
        agent = AgentDQN(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.SAC_V0:
        agent = AgentSAC(
            input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs, worker_id=worker_id,
            action_min=action_min, action_max=action_max, params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_A2C_V0:
        agent = AgentContinuousA2C(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            action_min=action_min, action_max=action_max, params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.DISCRETE_A2C_V0:
        agent = AgentDiscreteA2C(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_PPO_V0:
        agent = AgentContinuousPPO(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            action_min=action_min, action_max=action_max, params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.DISCRETE_PPO_V0:
        agent = AgentDiscretePPO(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.EVOLUTION_STRATEGY:
        agent = AgentEMAES(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.GENETIC_ALGORITHM:
        agent = AgentGA(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            params=params, device=device
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.MULTI_GENETIC_ALGORITHM:
        agent = AgentMultiGA(
            worker_id=worker_id, input_shape=input_shape, action_shape=action_shape, num_outputs=num_outputs,
            params=params, device=device
        )
    else:
        raise ValueError()

    return agent


def get_optimizer(parameters, learning_rate, params):
    if params.OPTIMIZER == OptimizerName.ADAM:
        optimizer = optim.Adam(params=parameters, lr=learning_rate, eps=1e-3)
    elif params.OPTIMIZER == OptimizerName.NESTEROV:
        optimizer = optim.SGD(params=parameters, lr=learning_rate, nesterov=True, momentum=0.9, weight_decay=1e-4)
    elif params.OPTIMIZER == OptimizerName.RMSProp:
        optimizer = torch.optim.RMSprop(params=parameters, lr=learning_rate, alpha=0.99, eps=1e-5)
    else:
        optimizer = None

    return optimizer
