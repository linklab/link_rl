import json

import gym
import paho.mqtt.client as mqtt
from torch import optim
import os, sys
import pybullet_envs

from codes.d_agents.continuous_action.continuous_a2c_agent import AgentContinuousA2C
from codes.d_agents.continuous_action.continuous_ppo_agent import AgentContinuousPPO
from codes.d_agents.discrete_action.discrete_a2c_agent import AgentDiscreteA2C
from codes.d_agents.discrete_action.dqn_agent import AgentDQN

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.b_environments.real_device.environment_rip import EnvironmentRIP
from codes.b_environments.unity.chaser_unity import Chaser_v1
from codes.b_environments.unity.drone_racing import Drone_Racing

from codes.c_models.continuous_action.deterministic_actor_critic_model import DeterministicActorCriticModel
from codes.c_models.continuous_action.stochastic_actor_critic_model import StochasticActorCriticModel
from codes.c_models.discrete_action.discrete_actor_critic_model import DiscreteActorCriticModel
from codes.c_models.discrete_action.dqn_model import DuelingDQNModel

from codes.d_agents.continuous_action.ddpg_agent import AgentDDPG

from codes.e_utils.actions import EpsilonGreedyDDPGActionSelector, EpsilonTracker, EpsilonGreedyDQNActionSelector, \
    ProbabilityActionSelector, ContinuousNormalActionSelector
from codes.e_utils.common_utils import make_atari_env
from codes.e_utils.names import EnvironmentName, DeepLearningModelName, RLAlgorithmName, OptimizerName


if params.MY_PLATFORM != "REAL_RIP_PLATFORM":
    from codes.b_environments.real_device.environment_double_rip import EnvironmentDoubleRIP

if params.MY_PLATFORM != "REAL_RIP_PLATFORM":
    from codes.b_environments.matlab.matlabenv import MatlabRotaryInvertedPendulumEnv

def get_environment(owner="chief", params=None):
    if params.ENVIRONMENT_ID == EnvironmentName.REAL_DEVICE_DOUBLE_RIP:
        env = EnvironmentDoubleRIP(
            owner=owner,
            action_min=params.ACTION_SCALE * -1.0,
            action_max=params.ACTION_SCALE,
            env_reset=params.ENV_RESET,
            params=params
        )

    elif params.ENVIRONMENT_ID == EnvironmentName.QUANSER_SERVO_2:
        client = mqtt.Client(client_id="env_sub_2", transport="TCP")
        env = EnvironmentRIP(mqtt_client=client)

        def __on_connect(client, userdata, flags, rc):
            print("mqtt broker connected with result code " + str(rc), flush=False)
            client.subscribe(topic=params.MQTT_SUB_FROM_SERVO)
            client.subscribe(topic=params.MQTT_SUB_MOTOR_LIMIT)
            client.subscribe(topic=params.MQTT_SUB_RESET_COMPLETE)

        def __on_log(client, userdata, level, buf):
            print(buf)

        def __on_message(client, userdata, msg):
            global PUB_ID

            if msg.topic == params.MQTT_SUB_FROM_SERVO:
                servo_info = json.loads(msg.payload.decode("utf-8"))
                motor_radian = float(servo_info["motor_radian"])
                motor_velocity = float(servo_info["motor_velocity"])
                pendulum_radian = float(servo_info["pendulum_radian"])
                pendulum_velocity = float(servo_info["pendulum_velocity"])
                pub_id = servo_info["pub_id"]
                env.set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

            elif msg.topic == params.MQTT_SUB_MOTOR_LIMIT:
                info = str(msg.payload.decode("utf-8")).split('|')
                pub_id = info[1]
                if info[0] == "limit_position":
                    env.is_motor_limit = True
                elif info[0] == "reset_complete":
                    env.is_limit_complete = True

            elif msg.topic == params.MQTT_SUB_RESET_COMPLETE:
                env.is_reset_complete = True
                servo_info = str(msg.payload.decode("utf-8")).split('|')
                motor_radian = float(servo_info[0])
                motor_velocity = float(servo_info[1])
                pendulum_radian = float(servo_info[2])
                pendulum_velocity = float(servo_info[3])
                pub_id = servo_info[4]
                env.set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

        if owner == "worker":
            client.on_connect = __on_connect
            client.on_message = __on_message
            # client.on_log = __on_log

            # client.username_pw_set(username="link", password="0123")
            client.connect(params.MQTT_SERVER_FOR_RIP, 1883, 3600)

            print("***** Sub thread started!!! *****", flush=False)
            client.loop_start()

    elif params.ENVIRONMENT_ID == EnvironmentName.CARTPOLE_V0:
        env = gym.make(EnvironmentName.CARTPOLE_V0.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.CARTPOLE_V1:
        env = gym.make(EnvironmentName.CARTPOLE_V1.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.CHASER_V1_MAC or params.ENVIRONMENT_ID == EnvironmentName.CHASER_V1_WINDOWS:
        env = Chaser_v1(params.MY_PLATFORM)
    elif params.ENVIRONMENT_ID in [EnvironmentName.BREAKOUT_DETERMINISTIC_V4, EnvironmentName.BREAKOUT_NO_FRAME_SKIP_V4, EnvironmentName.PONG_NO_FRAME_SKIP_V4, EnvironmentName.ENDURO_NO_FRAME_SKIP_V4, EnvironmentName.SEAQUEST_NO_FRAME_SKIP_V4, EnvironmentName.FREEWAY_NO_FRAME_SKIP_V4]:
        # env = gym.make(params.ENVIRONMENT_ID.value)
        env = make_atari_env(params.ENVIRONMENT_ID.value, seed=params.SEED)
        if params.SEED is not None:
            env.seed(params.SEED)
    elif params.ENVIRONMENT_ID == EnvironmentName.PENDULUM_V0:
        env = gym.make(EnvironmentName.PENDULUM_V0.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.ACROBOT_V1:
        env = gym.make(EnvironmentName.ACROBOT_V1.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.DRONE_RACING_MAC or params.ENVIRONMENT_ID == EnvironmentName.DRONE_RACING_WINDOWS:
        env = Drone_Racing(params.MY_PLATFORM)
    elif params.ENVIRONMENT_ID == EnvironmentName.BLACKJACK_V0:
        env = gym.make(EnvironmentName.BLACKJACK_V0.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.FROZENLAKE_V0:
        env = gym.make(EnvironmentName.FROZENLAKE_V0.value, is_slippery=False)
    elif params.ENVIRONMENT_ID == EnvironmentName.MOUNTAINCARCONTINUOUS_V0:
        env = gym.make(EnvironmentName.MOUNTAINCARCONTINUOUS_V0.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.INVERTED_DOUBLE_PENDULUM_V2:
        env = gym.make(EnvironmentName.INVERTED_DOUBLE_PENDULUM_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.HOPPER_V2:
        env = gym.make(EnvironmentName.HOPPER_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.ANT_V2:
        spec = gym.envs.registry.spec("AntBulletEnv-v0")
        spec._kwargs['render'] = params.ENV_RENDER
        env = gym.make("AntBulletEnv-v0")
    elif params.ENVIRONMENT_ID == EnvironmentName.SWIMMER_V2:
        env = gym.make(EnvironmentName.SWIMMER_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.HALF_CHEETAH_V2:
        spec = gym.envs.registry.spec("HalfCheetahBulletEnv-v0")
        spec._kwargs['render'] = params.ENV_RENDER
        env = gym.make("HalfCheetahBulletEnv-v0")
    elif params.ENVIRONMENT_ID == EnvironmentName.SWIMMER_V2:
        env = gym.make(EnvironmentName.SWIMMER_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.REACHER_V2:
        env = gym.make(EnvironmentName.REACHER_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.HUMANOID_V2:
        env = gym.make(EnvironmentName.HUMANOID_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.HUMANOID_STAND_UP_V2:
        env = gym.make(EnvironmentName.HUMANOID_STAND_UP_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.INVERTED_PENDULUM_V2:
        env = gym.make(EnvironmentName.INVERTED_PENDULUM_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.WALKER_2D_V2:
        env = gym.make(EnvironmentName.WALKER_2D_V2.value)
    elif params.ENVIRONMENT_ID == EnvironmentName.PENDULUM_MATLAB_V0:
        env = MatlabRotaryInvertedPendulumEnv(
            action_min=params.ACTION_SCALE * -1.0,
            action_max=params.ACTION_SCALE,
            env_reset=params.ENV_RESET,
            pendulum_type='PENDULUM_MATLAB_V0',
            params=params
        )
        env.start()
    elif params.ENVIRONMENT_ID == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
        env = MatlabRotaryInvertedPendulumEnv(
            action_min=params.ACTION_SCALE * -1.0,
            action_max=params.ACTION_SCALE,
            env_reset=params.ENV_RESET,
            pendulum_type='PENDULUM_MATLAB_DOUBLE_RIP_V0',
            params=params
        )
        env.start()
    elif params.ENVIRONMENT_ID == EnvironmentName.MINITAUR_BULLET_V0:
        import pybullet_envs
        spec = gym.envs.registry.spec("MinitaurBulletEnv-v0")
        spec._kwargs['render'] = params.ENV_RENDER
        env = gym.make("MinitaurBulletEnv-v0")
    else:
        env = None
    return env


def get_rl_model(worker_id, input_shape=None, num_outputs=None, params=None, device=None):
    if params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP:
        model = StochasticActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        )
    elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP:
        model = DiscreteActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP:
        model = DeterministicActorCriticModel(
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


def get_rl_agent(env, worker_id, params, device="cpu"):
    if params.RL_ALGORITHM == RLAlgorithmName.DDPG_FAST_V0:
        action_selector = EpsilonGreedyDDPGActionSelector(
            epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=params.ACTION_SCALE
        )

        epsilon_tracker = EpsilonTracker(
            action_selector=action_selector,
            eps_start=params.EPSILON_INIT,
            eps_final=params.EPSILON_MIN,
            eps_frames=params.EPSILON_MIN_STEP
        )

        input_shape = env.observation_space.shape
        num_outputs = env.action_space.shape[0]
        action_min = env.action_space.low[0]
        action_max = env.action_space.high[0]

        print("action_min: ", action_min, "action_max:", action_max)

        agent = AgentDDPG(
            input_shape=input_shape, num_outputs=num_outputs, worker_id=worker_id, action_selector=action_selector,
            action_min=action_min, action_max=action_max, params=params, device=device
        )

        return agent, epsilon_tracker
    elif params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_A2C_FAST_V0:
        action_selector = ContinuousNormalActionSelector()

        input_shape = env.observation_space.shape
        num_outputs = env.action_space.shape[0]
        action_min = env.action_space.low[0]
        action_max = env.action_space.high[0]

        print("action_min: ", action_min, "action_max:", action_max)

        agent = AgentContinuousA2C(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, action_selector=action_selector,
            action_min=action_min, action_max=action_max, params=params, device=device
        )

        return agent, None
    elif params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_PPO_FAST_V0:
        action_selector = ContinuousNormalActionSelector()

        input_shape = env.observation_space.shape
        num_outputs = env.action_space.shape[0]
        action_min = env.action_space.low[0]
        action_max = env.action_space.high[0]

        print("action_min: ", action_min, "action_max:", action_max)

        agent = AgentContinuousPPO(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, action_selector=action_selector,
            action_min=action_min, action_max=action_max, params=params, device=device
        )

        return agent, None
    elif params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
        action_selector = EpsilonGreedyDQNActionSelector(epsilon=params.EPSILON_INIT)

        epsilon_tracker = EpsilonTracker(
            action_selector=action_selector,
            eps_start=params.EPSILON_INIT,
            eps_final=params.EPSILON_MIN,
            eps_frames=params.EPSILON_MIN_STEP
        )

        input_shape = env.observation_space.shape
        num_outputs = env.action_space.n

        agent = AgentDQN(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs,
            action_selector=action_selector, params=params, device=device
        )

        return agent, epsilon_tracker
    elif params.RL_ALGORITHM == RLAlgorithmName.DISCRETE_A2C_FAST_V0:
        action_selector = ProbabilityActionSelector()

        input_shape = env.observation_space.shape
        num_outputs = env.action_space.n

        agent = AgentDiscreteA2C(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs,
            action_selector=action_selector, params=params, device=device
        )

        return agent, None
#
# def get_rl_algorithm(env, worker_id=0, logger=False, params=None):
#     if params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_A2C_FAST_V0:
#         rl_algorithm = CONTINUOUS_A2C_FAST_v0(
#             env=env,
#             worker_id=worker_id,
#             logger=logger,
#             params=params,
#             device=device,
#             verbose=params.VERBOSE
#         )
#     elif params.RL_ALGORITHM == RLAlgorithmName.DISCRETE_A2C_FAST_V0:
#         rl_algorithm = DISCRETE_A2C_FAST_v0(
#             env=env,
#             worker_id=worker_id,
#             logger=logger,
#             params=params,
#             device=device,
#             verbose=params.VERBOSE
#         )
#     elif params.RL_ALGORITHM == RLAlgorithmName.D4PG_FAST_V0:
#         rl_algorithm = D4PG_FAST_v0(
#             env=env,
#             worker_id=worker_id,
#             logger=logger,
#             params=params,
#             device=device,
#             verbose=params.VERBOSE
#         )
#     elif params.RL_ALGORITHM == RLAlgorithmName.DDPG_FAST_V0:
#         rl_algorithm = DDPG_FAST_v0(
#             env=env,
#             worker_id=worker_id,
#             logger=logger,
#             params=params,
#             device=device,
#             verbose=params.VERBOSE
#         )
#     else:
#         rl_algorithm = None
#
#     return rl_algorithm


def get_optimizer(parameters, learning_rate, params):
    if params.OPTIMIZER == OptimizerName.ADAM:
        optimizer = optim.Adam(params=parameters, lr=learning_rate)
    elif params.OPTIMIZER == OptimizerName.NESTEROV:
        optimizer = optim.SGD(params=parameters, lr=learning_rate, nesterov=True, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = None

    return optimizer
