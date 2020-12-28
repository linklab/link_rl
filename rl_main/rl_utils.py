import json
import threading

import gym
import paho.mqtt.client as mqtt
import torch
from torch import optim
import os, sys

from common.fast_rl.algorithms.D4PG_v0 import D4PG_FAST_v0
from common.fast_rl.algorithms.DUELING_DOUBLE_DQN_v0 import Dueling_Double_DQN_v0
from common.fast_rl.algorithms.PPO_v0 import PPO_FAST_v0
from common.models.dqn_model import DuelingDQNModel

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from config.parameters import PARAMETERS as params

if params.MY_PLATFORM != "REAL_RIP_PLATFORM":
    from common.environments.real_device.environment_double_rip import EnvironmentDoubleRIP

from common.fast_rl.algorithms.DDPG_RIP_DOUBLE_AGENTS_v0 import DDPG_FAST_RIP_DOUBLE_AGENTS_v0
from common.fast_rl.algorithms.DDPG_v0 import DDPG_FAST_v0

if params.MY_PLATFORM != "REAL_RIP_PLATFORM":
    from common.environments.matlab.matlabenv import MatlabRotaryInvertedPendulumEnv

from common.models.deterministic_actor_critic_model import DeterministicActorCriticModel
from config.names import EnvironmentName, DeepLearningModelName, RLAlgorithmName, OptimizerName

from common.environments.gym.frozenlake import FrozenLake_v0
from common.environments.gym.breakout import BreakoutDeterministic_v4
from common.environments.gym.cartpole import CartPole_v0, CartPole_v1
from common.environments.gym.pendulum import Pendulum_v0
from common.environments.gym.gridworld import GRIDWORLD_v0
from common.environments.gym.blackjack import Blackjack_v0
from common.environments.gym.mountaincar import MountainCarContinuous_v0
from common.environments.gym.acrobot import Acrobot_v1
from common.environments.real_device.environment_rip import EnvironmentRIP
from common.environments.unity.chaser_unity import Chaser_v1
from common.environments.unity.drone_racing import Drone_Racing
from common.environments.mujoco.inverted_double_pendulum import InvertedDoublePendulum_v2
from common.environments.mujoco.hopper import Hopper_v2
from common.environments.mujoco.ant import Ant_v2
from common.environments.mujoco.half_cheetah import HalfCheetah_v2
from common.environments.mujoco.swimmer import Swimmer_v2
from common.environments.mujoco.reacher import Reacher_v2
from common.environments.mujoco.humanoid import Humanoid_v2
from common.environments.mujoco.humanoid_stand_up import HumanoidStandUp_v2
from common.environments.mujoco.inverted_pendulum import InvertedPendulum_v2
from common.environments.mujoco.walker_2d import Walker2D_v2
from common.environments.real_device.environment_double_rip import EnvironmentDoubleRIP
from common.models.old_actor_critic_model import OldActorCriticModel
from common.algorithms_rl.DQN_v0 import DQN_v0
from common.algorithms_rl.Monte_Carlo_Control_v0 import Monte_Carlo_Control_v0
from common.algorithms_rl.PPO_v0 import PPO_v0
from common.algorithms_dp.DP_Policy_Iteration import Policy_Iteration
from common.algorithms_dp.DP_Value_Iteration import Value_Iteration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_environment(owner="chief", params=None):
    if params.ENVIRONMENT_ID == EnvironmentName.REAL_DEVICE_DOUBLE_RIP:
        client = mqtt.Client(client_id="env_pub_1", transport="TCP")
        env = EnvironmentDoubleRIP(
            action_min=params.SWING_UP_SCALE_FACTOR * -1.0,
            action_max=params.SWING_UP_SCALE_FACTOR,
            env_reset=params.ENV_RESET,
            mqtt_client = client
        )

        def __on_connect(client, userdata, flags, rc):
            print("mqtt broker connected with result code " + str(rc), flush=False)
            client.subscribe(topic=params.MQTT_SUB_FROM_DRIP)
            client.subscribe(topic=params.MQTT_SUB_RESET_COMPLETE)

        def __on_log(client, userdata, level, buf):
            print(buf)

        def __on_message(client, userdata, msg):
            global PUB_ID

            if msg.topic == params.MQTT_SUB_FROM_DRIP:
                servo_info = json.loads(msg.payload.decode("utf-8")).split('|')
                motor_position = float(servo_info[0])
                motor_velocity = float(servo_info[1])
                pendulum_position = float(servo_info[2])
                pendulum_velocity = float(servo_info[3])
                env.set_state(motor_position, motor_velocity, pendulum_position, pendulum_velocity)

            elif msg.topic == params.MQTT_SUB_RESET_COMPLETE:
                servo_info = str(msg.payload.decode("utf-8")).split('|')
                motor_position = float(servo_info[0])
                motor_velocity = float(servo_info[1])
                pendulum_position = float(servo_info[2])
                pendulum_velocity = float(servo_info[3])
                env.set_state(motor_position, motor_velocity, pendulum_position, pendulum_velocity)

        client.on_connect = __on_connect
        client.on_message = __on_message
        # client.on_log = __on_log
        #
        # # client.username_pw_set(username="link", password="0123")
        client.connect(params.MQTT_SERVER, 1883, 3600)
        #
        print("***** Sub thread started!!! *****", flush=False)
        client.loop_start()

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
        env = CartPole_v0()
    elif params.ENVIRONMENT_ID == EnvironmentName.CARTPOLE_V1:
        env = CartPole_v1()
    elif params.ENVIRONMENT_ID == EnvironmentName.CHASER_V1_MAC or params.ENVIRONMENT_ID == EnvironmentName.CHASER_V1_WINDOWS:
        env = Chaser_v1(params.MY_PLATFORM)
    elif params.ENVIRONMENT_ID == EnvironmentName.BREAKOUT_DETERMINISTIC_V4:
        env = BreakoutDeterministic_v4(params)
    elif params.ENVIRONMENT_ID == EnvironmentName.PENDULUM_V0:
        env = Pendulum_v0()
    elif params.ENVIRONMENT_ID == EnvironmentName.ACROBOT_V1:
        env = Acrobot_v1()
    elif params.ENVIRONMENT_ID == EnvironmentName.DRONE_RACING_MAC or params.ENVIRONMENT_ID == EnvironmentName.DRONE_RACING_WINDOWS:
        env = Drone_Racing(params.MY_PLATFORM)
    elif params.ENVIRONMENT_ID == EnvironmentName.GRIDWORLD_V0:
        env = GRIDWORLD_v0()
    elif params.ENVIRONMENT_ID == EnvironmentName.BLACKJACK_V0:
        env = Blackjack_v0()
    elif params.ENVIRONMENT_ID == EnvironmentName.FROZENLAKE_V0:
        env = FrozenLake_v0()
    elif params.ENVIRONMENT_ID == EnvironmentName.MOUNTAINCARCONTINUOUS_V0:
        env = MountainCarContinuous_v0()
    elif params.ENVIRONMENT_ID == EnvironmentName.INVERTED_DOUBLE_PENDULUM_V2:
        env = InvertedDoublePendulum_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.HOPPER_V2:
        env = Hopper_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.ANT_V2:
        env = Ant_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.HALF_CHEETAH_V2:
        env = HalfCheetah_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.SWIMMER_V2:
        env = Swimmer_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.REACHER_V2:
        env = Reacher_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.HUMANOID_V2:
        env = Humanoid_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.HUMANOID_STAND_UP_V2:
        env = HumanoidStandUp_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.INVERTED_PENDULUM_V2:
        env = InvertedPendulum_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.WALKER_2D_V2:
        env = Walker2D_v2()
    elif params.ENVIRONMENT_ID == EnvironmentName.PENDULUM_MATLAB_V0:
        env = MatlabRotaryInvertedPendulumEnv(
            action_min=params.SWING_UP_SCALE_FACTOR * -1.0,
            action_max=params.SWING_UP_SCALE_FACTOR,
            env_reset=params.ENV_RESET,
            pendulum_type= 'PENDULUM_MATLAB_V0'
        )
    elif params.ENVIRONMENT_ID == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
        env = MatlabRotaryInvertedPendulumEnv(
            action_min=params.SWING_UP_SCALE_FACTOR * -1.0,
            action_max=params.SWING_UP_SCALE_FACTOR,
            env_reset=params.ENV_RESET,
            pendulum_type= 'PENDULUM_MATLAB_DOUBLE_RIP_V0'
        )
    elif params.ENVIRONMENT_ID == EnvironmentName.MINITAUR_BULLET_V0:
        spec = gym.envs.registry.spec("MinitaurBulletEnv-v0")
        spec._kwargs['render'] = params.ENV_RENDER
        env = gym.make("MinitaurBulletEnv-v0")
    else:
        env = None
    return env


def get_rl_model(env, worker_id, params):
    if params.DEEP_LEARNING_MODEL == DeepLearningModelName.ACTOR_CRITIC_MLP:
        model = DeterministicActorCriticModel(
            s_size=env.n_states,
            a_size=env.n_actions,
            worker_id=worker_id,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL in [DeepLearningModelName.OLD_ACTOR_CRITIC_MLP, DeepLearningModelName.OLD_ACTOR_CRITIC_CNN]:
        model = OldActorCriticModel(
            s_size=env.n_states,
            a_size=env.n_actions,
            continuous=env.continuous,
            worker_id=worker_id,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL in [DeepLearningModelName.DUELING_DQN_MLP]:
        model = DuelingDQNModel(
            s_size=env.n_states,
            a_size=env.n_actions,
            worker_id=worker_id,
            params=params,
            device=device
        ).to(device)
    elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.NO_MODEL:
        model = None
    else:
        model = None
    return model


def get_rl_algorithm(env, worker_id=0, logger=False, params=None):
    if params.RL_ALGORITHM == RLAlgorithmName.D4PG_FAST_V0:
        rl_algorithm = D4PG_FAST_v0(
            env=env,
            worker_id=worker_id,
            logger=logger,
            params=params,
            device=device,
            verbose=params.VERBOSE
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.DDPG_FAST_V0:
        rl_algorithm = DDPG_FAST_v0(
            env=env,
            worker_id=worker_id,
            logger=logger,
            params=params,
            device=device,
            verbose=params.VERBOSE
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.PPO_FAST_V0:
        rl_algorithm = PPO_FAST_v0(
            env=env,
            worker_id=worker_id,
            logger=logger,
            params=params,
            device=device,
            verbose=params.VERBOSE
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.PPO_V0:
        rl_algorithm = PPO_v0(
            env=env,
            worker_id=worker_id,
            gamma=params.GAMMA,
            env_render=params.ENV_RENDER,
            logger=logger,
            params=params,
            device=device,
            verbose=params.VERBOSE
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
        rl_algorithm = Dueling_Double_DQN_v0(
            env=env,
            worker_id=worker_id,
            logger=logger,
            params=params,
            device=device,
            verbose=params.VERBOSE
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.DQN_V0:
        rl_algorithm = DQN_v0(
            env=env,
            worker_id=worker_id,
            gamma=params.GAMMA,
            env_render=params.ENV_RENDER,
            logger=logger,
            params=params,
            device=device,
            verbose=params.VERBOSE
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.Policy_Iteration:
        rl_algorithm = Policy_Iteration(
            env=env,
            gamma=params.GAMMA,
            params=params
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.Value_Iteration:
        rl_algorithm = Value_Iteration(
            env=env,
            gamma=params.GAMMA,
            params=params
        )
    elif params.RL_ALGORITHM == RLAlgorithmName.Monte_Carlo_Control_V0:
        rl_algorithm = Monte_Carlo_Control_v0(
            env=env,
            worker_id=worker_id,
            gamma=params.GAMMA,
            env_render=params.ENV_RENDER,
            logger=logger,
            params=params,
            verbose=params.VERBOSE
        )
    else:
        rl_algorithm = None

    return rl_algorithm


def get_optimizer(parameters, learning_rate, params):
    if params.OPTIMIZER == OptimizerName.ADAM:
        optimizer = optim.Adam(params=parameters, lr=learning_rate)
    elif params.OPTIMIZER == OptimizerName.NESTEROV:
        optimizer = optim.SGD(params=parameters, lr=learning_rate, nesterov=True, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = None

    return optimizer
