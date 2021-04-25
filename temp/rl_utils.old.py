import json
import random

from gym import Env
from gym.spaces import Box, Discrete
from gym.vector import SyncVectorEnv, VectorEnv
from numpy import random

import gym
import paho.mqtt.client as mqtt
from torch import optim
import os, sys

from codes.c_models.continuous_action.soft_actor_critic_model import SoftActorCriticModel
from codes.d_agents.off_policy.sac.continuous_sac_agent import AgentSAC

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.d_agents.on_policy.a2c.continuous_a2c_agent import AgentContinuousA2C
from codes.d_agents.on_policy.ppo.continuous_ppo_agent import AgentContinuousPPO
from codes.d_agents.on_policy.a2c.discrete_a2c_agent import AgentDiscreteA2C
from codes.d_agents.off_policy.dqn.dqn_agent import AgentDQN

from codes.c_models.continuous_action.deterministic_continuous_actor_critic_model import DeterministicActorCriticModel
from codes.c_models.continuous_action.stochastic_continuous_actor_critic_model import StochasticActorCriticModel
from codes.c_models.discrete_action.discrete_actor_critic_model import DiscreteActorCriticModel
from codes.c_models.discrete_action.dqn_model import DuelingDQNModel

from codes.d_agents.off_policy.ddpg.ddpg_agent import AgentDDPG

from codes.d_agents.actions import EpsilonGreedyDDPGActionSelector, EpsilonTracker, EpsilonGreedyDQNActionSelector, \
    ProbabilityActionSelector, ContinuousNormalActionSelector, EpsilonGreedySomeTimesBlowDDPGActionSelector
from codes.e_utils.common_utils import make_atari_env
from codes.e_utils.names import EnvironmentName, DeepLearningModelName, RLAlgorithmName, OptimizerName


def get_environment(params):
    def make_environment(params):
        def _make():
            env = get_single_environment(params=params)
            return env

        return _make
    env_fns = [make_environment(params=params) for _ in range(params.NUM_ENVIRONMENTS)]
    env = SyncVectorEnv(env_fns)
    assert env.num_envs == params.NUM_ENVIRONMENTS
    return env


def get_single_environment(owner="cheif", params=None):
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
        from codes.b_environments.quanser_rotary_inverted_pendulum.old.quanser_rip import EnvironmentRIP
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
    elif params.ENVIRONMENT_ID in [
        EnvironmentName.CARTPOLE_V0, EnvironmentName.CARTPOLE_V1, EnvironmentName.PENDULUM_V0,
        EnvironmentName.ACROBOT_V1, EnvironmentName.BLACKJACK_V0, EnvironmentName.MOUNTAINCARCONTINUOUS_V0,
        EnvironmentName.INVERTED_DOUBLE_PENDULUM_V2, EnvironmentName.HOPPER_V2, EnvironmentName.SWIMMER_V2,
        EnvironmentName.REACHER_V2, EnvironmentName.HUMANOID_V2, EnvironmentName.HUMANOID_STAND_UP_V2,
        EnvironmentName.INVERTED_PENDULUM_V2, EnvironmentName.WALKER_2D_V2,
    ]:
        env = gym.make(params.ENVIRONMENT_ID.value)
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
        EnvironmentName.FREEWAY_NO_FRAME_SKIP_V4
    ]:
        env = make_atari_env(params.ENVIRONMENT_ID.value)
    elif params.ENVIRONMENT_ID in [EnvironmentName.DRONE_RACING_MAC, EnvironmentName.DRONE_RACING_WINDOWS]:
        from codes.b_environments.unity.drone_racing import Drone_Racing
        env = Drone_Racing(params.MY_PLATFORM)
    elif params.ENVIRONMENT_ID in [
        EnvironmentName.PYBULLET_ANT_V0, EnvironmentName.PYBULLET_HALF_CHEETAH_V0,
        EnvironmentName.PYBULLET_MINITAUR_BULLET_V0, EnvironmentName.PYBULLET_INVERTED_DOUBLE_PENDULUM_V0
    ]:
        spec = gym.envs.registry.spec(params.ENVIRONMENT_ID.value)
        spec._kwargs['render'] = params.ENV_RENDER
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
        if isinstance(env.single_action_space, Discrete):
            num_outputs = env.single_action_space.n
            action_min, action_max = None, None
        elif isinstance(env.single_action_space, Box):
            num_outputs = env.single_action_space.shape[0]
            action_min = env.single_action_space.low[0]
            action_max = env.single_action_space.high[0]
        else:
            num_outputs, action_min, action_max = None, None, None
    elif isinstance(env, Env):
        input_shape = env.observation_space.shape
        if isinstance(env.action_space, Discrete):
            num_outputs = env.action_space.n
            action_min, action_max = None, None
        elif isinstance(env.action_space, Box):
            num_outputs = env.action_space.shape[0]
            action_min = env.action_space.low[0]
            action_max = env.action_space.high[0]
        else:
            num_outputs, action_min, action_max = None, None, None
    else:
        raise ValueError()

    print(f"num_outputs: {num_outputs}, action_min: {action_min}, action_max: {action_max}")

    return input_shape, num_outputs, action_min, action_max


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
        model = StochasticActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
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
    input_shape, num_outputs, action_min, action_max = get_environment_input_output_info(env)

    if params.RL_ALGORITHM == RLAlgorithmName.DDPG_V0:
        if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0]:
            action_selector = EpsilonGreedySomeTimesBlowDDPGActionSelector(
                epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=params.ACTION_SCALE,
                min_blowing_action=-10.0 * params.ACTION_SCALE, max_blowing_action=10.0 * params.ACTION_SCALE
            )
        else:
            action_selector = EpsilonGreedyDDPGActionSelector(
                epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=params.ACTION_SCALE
            )

        epsilon_tracker = EpsilonTracker(
            action_selector=action_selector,
            eps_start=params.EPSILON_INIT,
            eps_final=params.EPSILON_MIN,
            eps_frames=params.EPSILON_MIN_STEP
        )

        agent = AgentDDPG(
            input_shape=input_shape, num_outputs=num_outputs, worker_id=worker_id, action_selector=action_selector,
            action_min=action_min, action_max=action_max, params=params, device=device
        )

        return agent, epsilon_tracker
    elif params.RL_ALGORITHM == RLAlgorithmName.SAC_V0:
        action_selector = ContinuousNormalActionSelector()

        agent = AgentSAC(
            input_shape=input_shape, num_outputs=num_outputs, worker_id=worker_id, action_selector=action_selector,
            action_min=action_min, action_max=action_max, params=params, device=device
        )

        return agent, None
    elif params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_A2C_V0:
        action_selector = ContinuousNormalActionSelector()

        agent = AgentContinuousA2C(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, action_selector=action_selector,
            action_min=action_min, action_max=action_max, params=params, device=device
        )

        return agent, None
    elif params.RL_ALGORITHM == RLAlgorithmName.CONTINUOUS_PPO_V0:
        action_selector = ContinuousNormalActionSelector()

        agent = AgentContinuousPPO(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, action_selector=action_selector,
            action_min=action_min, action_max=action_max, params=params, device=device
        )

        return agent, None
    elif params.RL_ALGORITHM == RLAlgorithmName.DQN_V0:
        action_selector = EpsilonGreedyDQNActionSelector(epsilon=params.EPSILON_INIT)

        epsilon_tracker = EpsilonTracker(
            action_selector=action_selector,
            eps_start=params.EPSILON_INIT,
            eps_final=params.EPSILON_MIN,
            eps_frames=params.EPSILON_MIN_STEP
        )

        agent = AgentDQN(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs,
            action_selector=action_selector, params=params, device=device
        )

        return agent, epsilon_tracker
    elif params.RL_ALGORITHM == RLAlgorithmName.DISCRETE_A2C_V0:
        action_selector = ProbabilityActionSelector()

        agent = AgentDiscreteA2C(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs,
            action_selector=action_selector, params=params, device=device
        )

        return agent, None


def get_optimizer(parameters, learning_rate, params):
    if params.OPTIMIZER == OptimizerName.ADAM:
        optimizer = optim.Adam(params=parameters, lr=learning_rate)
    elif params.OPTIMIZER == OptimizerName.NESTEROV:
        optimizer = optim.SGD(params=parameters, lr=learning_rate, nesterov=True, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = None

    return optimizer
