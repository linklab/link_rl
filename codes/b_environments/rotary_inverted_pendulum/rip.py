import math
import random

import grpc
import gym
import numpy as np
import time
import sys,os
import matplotlib.pyplot as plt
import pickle
from codes.a_config.parameters_general import RIPEnvRewardType
from codes.e_utils.common_utils import sigmoid_2

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.b_environments.rotary_inverted_pendulum import rip_service_pb2_grpc
from codes.b_environments.rotary_inverted_pendulum.rip_service_pb2 import RipRequest
from codes.b_environments.rotary_inverted_pendulum.matlabcode import SimulinkPlant

from codes.e_utils.names import RLAlgorithmName, EnvironmentName

from gym.envs.classic_control.acrobot import wrap

from codes.a_config.parameters import PARAMETERS as params

np.set_printoptions(formatter={'float_kind': lambda x: '{0:0.6f}'.format(x)})

BLOWING_ACTION_RATE = 0.0002  # 5000 스텝에 1번 정도(지수 분포)의 주가로 외력이 가해짐 --> Stochastic Env.
# BLOWING_ACTION_RATE = 0.000000000002

VELOCITY_STATE_DENOMINATOR = 100.0


if params.SERVER_IDX == 0:
    RIP_SERVER = '10.0.0.9'
elif params.SERVER_IDX ==1:
    RIP_SERVER = '10.0.0.11'

def get_rip_observation_space(pendulum_type, params):
    max_velocity = 100.0

    if pendulum_type in [
        EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.REAL_DEVICE_RIP
    ]:
        high = np.array(
            [1., 1., max_velocity, 1., 1., max_velocity],
            dtype=np.float32
        )
    elif pendulum_type in [
        EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP
    ]:
        if hasattr(params, "IGNORE_ARM_INFO") and params.IGNORE_ARM_INFO is True:
            high = np.array(
                [1., 1., max_velocity, 1., 1., max_velocity],
                dtype=np.float32
            )
        else:
            high = np.array(
                [1., 1., max_velocity, 1., 1., max_velocity, 1., 1., max_velocity],
                dtype=np.float32
            )
    else:
        raise ValueError()

    low = high * -1.0

    observation_space = gym.spaces.Box(
        low=low, high=high, dtype=np.float32
    )

    n_states = observation_space.shape[0]

    return observation_space, n_states


def get_rip_action_space(params, pendulum_type):
    if pendulum_type == EnvironmentName.PENDULUM_MATLAB_V0:
        action_index_to_voltage = [
            -0.08, -0.05, -0.025, -0.0125, -0.008, -0.002, 0.0, 0.002, 0.008, 0.0125, 0.025, 0.05, 0.08
        ]
    elif pendulum_type == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
        action_index_to_voltage = [
            -3.5, -2.75, -2.0, -1.5, -0.75, -0.35, -0.10, -0.05, -0.025, -0.016, 0.0,
            0.016, 0.025, 0.05, 0.10, 0.35, 0.75, 1.5, 2.0, 2.75, 3.5
        ]
    elif pendulum_type in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
        action_index_to_voltage = None
    else:
        raise ValueError()

    if params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
        action_space = gym.spaces.Discrete(len(action_index_to_voltage))
        n_actions = action_space.n
    else:
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,),
            dtype=np.float32
        )
        n_actions = action_space.shape[0]

    return action_space, n_actions, action_index_to_voltage


class RotaryInvertedPendulumEnv(gym.Env):
    def __init__(
            self, action_min, action_max, env_reset=True,
            pendulum_type=EnvironmentName.PENDULUM_MATLAB_V0, params=None
    ):
        self.episode_steps = 0
        self.total_steps = 0
        self.env_reset = env_reset
        self.pendulum_type = pendulum_type
        self.action_min = action_min
        self.action_max = action_max
        self.params = params

        self.pendulum_1_position = 0
        self.pendulum_1_velocity = 0
        self.pendulum_2_position = 0
        self.pendulum_2_velocity = 0
        self.motor_position = 0
        self.motor_velocity = 0

        self.last_time = 0.0
        self.unit_time = 0.006
        # self.unit_time = 0.06
        self.over_unit_time = 0
        self.step_idx = 0
        self.episode_idx = 0

        current_path = os.path.dirname(os.path.realpath(__file__))
        MATLAB_ENGINE_DIR = os.path.abspath(os.path.join(current_path, "engine"))
        os.chdir(MATLAB_ENGINE_DIR) # change working directory

        if self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_V0:
            self.plant = SimulinkPlant(modelName="single_RIP")
        elif self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
            self.plant = SimulinkPlant(modelName="double_RIP")
        elif self.pendulum_type in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            self.plant = None
        else:
            raise ValueError()

        self.obs_degree = [None, None]
        self.next_obs_degree = [None, None]
        self.simulation_time = 0.0

        self.too_much_rotate = False

        self.action_space, self.n_actions, self.action_index_to_voltage = get_rip_action_space(
            params, self.pendulum_type
        )

        self.observation_space, self.n_states = get_rip_observation_space(self.pendulum_type, self.params)

        self.current_status = None

        self.count_continuous_uprights = 0
        self.is_upright = False
        self.initial_motor_position = 0.0

        # self.episode_position_reward_list = []
        # self.episode_pendulum_velocity_reward_list = []
        # self.episode_action_reward_list = []

        self.next_time_step_of_external_blow = int(random.expovariate(BLOWING_ACTION_RATE))

        self.num_episodes = 0
        self.episode_period_env_reset_forced = 10

        if self.pendulum_type in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            channel = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER))
            self.server_obj = rip_service_pb2_grpc.RDIPStub(channel)
        else:
            self.server_obj = None

        self.max_pendulum_1_velocity = 0.0
        self.max_pendulum_2_velocity = 0.0
        self.max_motor_velocity = 0.0

    def get_n_states(self):
        n_states = self.observation_space.shape[0]
        return n_states

    def get_n_actions(self):
        if self.params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
            n_actions = self.action_space.n
        else:
            n_actions = self.action_space.shape[0]
        return n_actions

    @property
    def action_meanings(self):
        if self.params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
            action_meanings = self.action_index_to_voltage
        else:
            action_meanings = ["Joint effort",]
        return action_meanings

    def pause(self):
        self.plant.conncectpause()

    def start(self):
        self.plant.connectToMatlab()

    def disconnect(self):
        self.plant.disconnect()

    def reset(self):
        self.episode_steps = 0
        self.episode_idx += 1

        if self.total_steps == 0:
            print("next_time_step_of_external_blow: {0}".format(
                self.next_time_step_of_external_blow
            ))

        if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_V0,EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0]:
            if self.env_reset or self.num_episodes % self.episode_period_env_reset_forced == 0:
                print("ENV RESET")
                self.plant.connectStart()
        # else:
        #     if self.env_reset or self.num_episodes % self.episode_period_env_reset_forced == 0:
        #         print("ENV RESET")
        #         #TODO : 리셋할때 뭐 할지 코

        # self.episode_position_reward_list.clear()
        # self.episode_pendulum_velocity_reward_list.clear()
        # self.episode_action_reward_list.clear()

        if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.REAL_DEVICE_RIP]:
            # self.update_current_state(adjusted_pendulum_1_radian=0.0)
            if self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_V0:
                self.pendulum_1_position, self.motor_position, self.pendulum_1_velocity, self.motor_velocity, _  \
                    = self.plant.getHistory
            else:
                rip_response = self.server_obj.reset(RipRequest(value=None))

                self.motor_position = math.radians(rip_response.arm_angle)
                self.motor_velocity = rip_response.arm_velocity
                self.pendulum_1_position = math.radians(rip_response.link_1_angle)
                self.pendulum_1_velocity = rip_response.link_1_velocity
                self.simulation_time = None

            state = (
                math.cos(self.pendulum_1_position),
                math.sin(self.pendulum_1_position),
                self.pendulum_1_velocity,
                math.cos(0.0),  # 1.0
                math.sin(0.0),  # 0.0
                self.motor_velocity,
            )

            self.update_current_state_for_double_rip(adjusted_pendulum_1_radian=0.0, adjusted_pendulum_2_radian=0.0)

        elif self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            if self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
                self.pendulum_1_position, self.motor_position, self.pendulum_2_position, self.pendulum_1_velocity, \
                self.motor_velocity, self.pendulum_2_velocity, self.simulation_time = self.plant.getHistory()
            else:
                rip_response = self.server_obj.reset(RipRequest(value=None))

                self.motor_position = math.radians(rip_response.arm_angle)
                self.motor_velocity = rip_response.arm_velocity
                self.pendulum_1_position = math.radians(rip_response.link_1_angle)
                self.pendulum_1_velocity = rip_response.link_1_velocity
                self.pendulum_2_position = math.radians(rip_response.link_2_angle)
                self.pendulum_2_velocity = rip_response.link_2_velocity
                self.simulation_time = None

            self.set_max_three_velocity()

            if hasattr(self.params, "IGNORE_ARM_INFO") and self.params.IGNORE_ARM_INFO is True:
                state = (
                    math.cos(self.pendulum_1_position),
                    math.sin(self.pendulum_1_position),
                    self.pendulum_1_velocity / VELOCITY_STATE_DENOMINATOR,
                    math.cos(self.pendulum_2_position),
                    math.sin(self.pendulum_2_position),
                    self.pendulum_2_velocity / VELOCITY_STATE_DENOMINATOR,
                )
            else:
                state = (
                    math.cos(self.pendulum_1_position),
                    math.sin(self.pendulum_1_position),
                    self.pendulum_1_velocity / VELOCITY_STATE_DENOMINATOR,
                    math.cos(self.pendulum_2_position),
                    math.sin(self.pendulum_2_position),
                    self.pendulum_2_velocity / VELOCITY_STATE_DENOMINATOR,
                    math.cos(0.0),  # 1.0
                    math.sin(0.0),  # 0.0
                    self.motor_velocity / VELOCITY_STATE_DENOMINATOR,
                )

            self.update_current_state_for_double_rip(adjusted_pendulum_1_radian=0.0, adjusted_pendulum_2_radian=0.0)
        else:
            raise ValueError()

        # print("ENV RESET")
        # print(state)

        self.too_much_rotate = False

        self.count_continuous_uprights = 0
        self.is_upright = False

        self.initial_motor_position = self.motor_position

        return state

    def set_max_three_velocity(self):
        if self.pendulum_1_velocity > self.max_pendulum_1_velocity:
            self.max_pendulum_1_velocity = self.pendulum_1_velocity

        if self.pendulum_2_velocity > self.max_pendulum_2_velocity:
            self.max_pendulum_2_velocity = self.pendulum_2_velocity

        if self.motor_velocity > self.max_motor_velocity:
            self.max_motor_velocity = self.motor_velocity

    @staticmethod
    def pendulum_position_to_adjusted_radian(position):
        # radian을 0과 2 * math.pi 사이 값(양수)으로 조정
        if abs(position) > 2 * math.pi:
            q_ = abs(position) % (2 * math.pi)
        else:
            q_ = abs(position)

        # radian을 0과 math.pi 사이 값(양수)으로 조정: 3 * math.pi / 2 -->  2 * math.pi - 3 * math.pi / 2 --> math.pi / 2
        if q_ > math.pi:
            adjusted_radian = 2 * math.pi - q_
        else:
            adjusted_radian = q_

        return adjusted_radian

    def update_current_state(self, adjusted_pendulum_1_radian):
        if math.pi - math.radians(12) < adjusted_pendulum_1_radian <= math.pi:
            self.count_continuous_uprights += 1
        else:
            self.count_continuous_uprights = 0

        if self.count_continuous_uprights >= 1:
            self.is_upright = True
        else:
            self.is_upright = False

    def update_current_state_for_double_rip(self, adjusted_pendulum_1_radian, adjusted_pendulum_2_radian):
        if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            upright_conditions = [
                math.pi - math.radians(12) < adjusted_pendulum_1_radian <= math.pi,
                adjusted_pendulum_2_radian < math.radians(12)
            ]
        else:
            upright_conditions = [
                math.pi - math.radians(12) < adjusted_pendulum_1_radian <= math.pi,
                #math.pi - math.radians(12) < adjusted_pendulum_2_radian <= math.pi
            ]

        if all(upright_conditions):
            self.count_continuous_uprights += 1
        else:
            self.count_continuous_uprights = 0

        if self.count_continuous_uprights >= 1:
            self.is_upright = True
        else:
            self.is_upright = False

    def step(self, action):
        #action = 100
        ############# time check #############################
        if self.pendulum_type in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            while True:
                current_time = time.perf_counter()
                if current_time - self.last_time >= self.unit_time:
                    break
                time.sleep(0.0001)

            current_time = time.perf_counter()
            step_time = current_time - self.last_time

            if step_time > self.unit_time:
                self.over_unit_time += 1

            # print(self.step_idx, action, step_time)
            self.last_time = time.perf_counter()

            if self.step_idx % 100000 == 0:
                print("*OVER UNIT TIME STEP NUMBER :", self.over_unit_time)
            #######################################################

        self.episode_steps += 1
        self.total_steps += 1
        # print("action", action, "total steps", self.total_steps, "episode steps", self.episode_steps)

        if self.total_steps >= self.next_time_step_of_external_blow:
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
                action = random.uniform(
                    a=self.action_index_to_voltage[0] * 10.0,
                    b=self.action_index_to_voltage[-1] * 10.0,
                )
            elif self.params.RL_ALGORITHM in [
                RLAlgorithmName.DDPG_V0,
                RLAlgorithmName.CONTINUOUS_A2C_V0,
                RLAlgorithmName.CONTINUOUS_PPO_V0,
                RLAlgorithmName.TD3_V0,
                RLAlgorithmName.SAC_V0
            ]:
                action = random.uniform(a=-1.0, b=1.0) * self.params.ACTION_SCALE * 2
            else:
                raise ValueError()

            self.next_time_step_of_external_blow = self.total_steps + int(random.expovariate(BLOWING_ACTION_RATE))
            print("[{0:6}/{1}] External Blow: {2:7.5f}, next_time_step_of_external_blow: {3}".format(
                self.step_idx + 1,
                self.params.MAX_GLOBAL_STEP,
                action,
                self.next_time_step_of_external_blow
            ))
        else:
            if type(action) is np.ndarray:
                action = action[0]

            if self.params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
                action = self.action_index_to_voltage[action]

        if self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_V0:
            self.plant.simulate(action)

            self.pendulum_1_position, self.motor_position, self.pendulum_1_velocity, self.motor_velocity, \
            self.simulation_time = self.plant.getHistory()
        elif self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
            self.plant.simulate(action)

            self.pendulum_1_position, self.motor_position, self.pendulum_2_position, self.pendulum_1_velocity, \
            self.motor_velocity, self.pendulum_2_velocity, self.simulation_time = self.plant.getHistory()
        elif self.pendulum_type == EnvironmentName.REAL_DEVICE_RIP:
            # GRPC CALL
            rip_response = self.server_obj.step(RipRequest(value=action))
            # current_time = time.perf_counter()
            # print("point 2 - elapsed time: {0:10.8f}".format(current_time - self.last_time))
            self.motor_position = math.radians(rip_response.arm_angle)
            self.motor_velocity = rip_response.arm_velocity
            self.pendulum_1_position = math.radians(rip_response.link_1_angle)
            self.pendulum_1_velocity = rip_response.link_1_velocity
            self.simulation_time = None
            # print(rip_response.link_1_angle)
        elif self.pendulum_type == EnvironmentName.REAL_DEVICE_DOUBLE_RIP:
            # t = 0
            # num = 0
            # k = 1
            # while True:
            #     pwm = 200
            #     # if num % 300 == 0:
            #     #     pwm += 1
            #
            #     self.server_obj.step(RipRequest(value=pwm))
            #
            #     # action = 200 * math.sin(2*0.1*math.pi*t)
            #     # self.server_obj.step(RipRequest(value=action))
            #     print("{0:4d} {1:>6.4f} {2:>5.1f}".format(num, t, pwm))
            #     t += 0.008
            #     time.sleep(0.008)
            #     num += 1

            # print(action, self.step_idx)
            rip_response = self.server_obj.step(RipRequest(value=action))
            # print(action, rip_response.arm_angle, rip_response.link_1_angle, "!!!!")

            self.motor_position = math.radians(rip_response.arm_angle)
            self.motor_velocity = rip_response.arm_velocity
            self.pendulum_1_position = math.radians(rip_response.link_1_angle)
            self.pendulum_1_velocity = rip_response.link_1_velocity
            self.pendulum_2_position = math.radians(rip_response.link_2_angle)
            self.pendulum_2_velocity = rip_response.link_2_velocity
            self.simulation_time = None

        else:
            raise ValueError()

        #print(self.motor_position, math.cos(self.motor_position), math.sin(self.motor_position))
        # print("!!!!!!!!!", self.pendulum_2_position)

        if abs(self.initial_motor_position - self.motor_position) > math.pi * 3:
            self.too_much_rotate = True
        # print(self.initial_motor_position, self.motor_position)
        done_conditions = [
            self.episode_steps >= self.params.MAX_EPISODE_STEP, #5000
            # self.episode_steps >= self.params.MAX_EPISODE_STEP and not self.is_upright,
            self.too_much_rotate and not self.is_upright
        ]

        adjusted_pendulum_1_radian = self.pendulum_position_to_adjusted_radian(self.pendulum_1_position)
        adjusted_pendulum_2_radian = self.pendulum_position_to_adjusted_radian(self.pendulum_2_position)
        # print("pendulum 1 angle :", adjusted_pendulum_1_radian, "pendulum 2 angle :", adjusted_pendulum_2_radian)

        if self.pendulum_type in [
            EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.REAL_DEVICE_RIP
        ]:
            self.update_current_state(adjusted_pendulum_1_radian)
            reward = self.get_reward(adjusted_pendulum_1_radian)
        elif self.pendulum_type in [
            EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP
        ]:
            self.update_current_state_for_double_rip(adjusted_pendulum_1_radian, adjusted_pendulum_2_radian)

            if self.params.TYPE_OF_RIP_REWARD == RIPEnvRewardType.NEW:
                reward = self.get_reward_for_double_rip_1()
            elif self.params.TYPE_OF_RIP_REWARD == RIPEnvRewardType.OLD:
                reward = self.get_reward_for_double_rip_2()
            elif self.params.TYPE_OF_RIP_REWARD == RIPEnvRewardType.UNTIL_TERMINAL_ZERO:
                reward = self.get_reward_for_double_rip_3()
            elif self.params.TYPE_OF_RIP_REWARD == RIPEnvRewardType.ORIGINAL:
                reward = self.get_reward_for_double_rip_4(self.pendulum_1_position, self.pendulum_2_position)
            else:
                raise ValueError()
                # print("REWARD :", reward)
        else:
            raise ValueError()
        # print(done_conditions)
        if any(done_conditions):
            done = True

            info = {
                "adjusted_pendulum_1_radian": adjusted_pendulum_1_radian,
                "adjusted_pendulum_2_radian": adjusted_pendulum_2_radian if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP] else None,
                # "episode_position_reward_list": sum(self.episode_position_reward_list),
                # "episode_pendulum_velocity_reward": sum(self.episode_pendulum_velocity_reward_list),
                # "episode_action_reward": sum(self.episode_action_reward_list)
            }

            self.num_episodes += 1

            if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.PENDULUM_MATLAB_V0]:
                if self.env_reset or self.num_episodes % self.episode_period_env_reset_forced == 0:
                    self.plant.connectStop()

        else:
            done = False
            info = {
                "adjusted_pendulum_1_radian": adjusted_pendulum_1_radian,
                "adjusted_pendulum_2_radian": adjusted_pendulum_2_radian if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP] else None
            }

        if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.REAL_DEVICE_RIP]:
            state = (
                math.cos(self.pendulum_1_position),
                math.sin(self.pendulum_1_position),
                self.pendulum_1_velocity,
                math.cos(self.initial_motor_position - self.motor_position),
                math.sin(self.initial_motor_position - self.motor_position),
                self.motor_velocity,
            )
        elif self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            self.set_max_three_velocity()

            # info["max_pendulum_1_velocity"] = self.max_pendulum_1_velocity
            # info["max_pendulum_2_velocity"] = self.max_pendulum_2_velocity
            # info["max_motor_velocity"] = self.max_motor_velocity
            if hasattr(self.params, "IGNORE_ARM_INFO") and self.params.IGNORE_ARM_INFO is True:
                state = (
                    math.cos(self.pendulum_1_position),
                    math.sin(self.pendulum_1_position),
                    self.pendulum_1_velocity / VELOCITY_STATE_DENOMINATOR,
                    math.cos(self.pendulum_2_position),
                    math.sin(self.pendulum_2_position),
                    self.pendulum_2_velocity / VELOCITY_STATE_DENOMINATOR,
                )
            else:
                state = (
                    math.cos(self.pendulum_1_position),
                    math.sin(self.pendulum_1_position),
                    self.pendulum_1_velocity / VELOCITY_STATE_DENOMINATOR,
                    math.cos(self.pendulum_2_position),
                    math.sin(self.pendulum_2_position),
                    self.pendulum_2_velocity / VELOCITY_STATE_DENOMINATOR,
                    math.cos(self.initial_motor_position - self.motor_position),
                    math.sin(self.initial_motor_position - self.motor_position),
                    self.motor_velocity / VELOCITY_STATE_DENOMINATOR,
                )

            # print("STEP: {0}, ACTION: {1:>6.2f}, pendulum_1_velocity: {2:>5.2f}, pendulum_2_velocity: {3:>5.2f}, "
            #       "motor_velocity: {4:>5.2f}, {5:>5.2f}, {6:>5.2f}, {7:>5.2f} - reward: {8:>5.2f}".format(
            #     self.step_idx, action, self.pendulum_1_velocity, self.pendulum_2_velocity, self.motor_velocity,
            #     self.pendulum_1_position, self.pendulum_2_position, self.motor_position, reward
            # ))

            # print("pendulum_2 :", self.pendulum_2_position)
        else:
            raise ValueError()
        # print(adjusted_pendulum_1_radian, reward)
        # time.sleep(0.5)
        self.step_idx += 1
        # print(self.episode_steps, done, "!!!!!!")

        # if self.num_episodes % 3 == 0 and self.episode_steps == 1:
        #     print("PENDULUM 1 : {0:7.4f}, PENDULUM 2 : {1:7.4f}".format(adjusted_pendulum_1_radian, adjusted_pendulum_2_radian))

        # print(action)

        if self.pendulum_type in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            info["unit_time"] = self.unit_time


        return state, reward, done, info

    def get_reward(self, adjusted_pendulum_1_radian):
        if self.is_upright:
            position_reward = adjusted_pendulum_1_radian # math.pi - math.radians(12) ~ math.pi
        else:
            position_reward = adjusted_pendulum_1_radian / 2.0

        energy_penalty = -1.0 * (abs(self.pendulum_1_velocity) + abs(self.motor_velocity)) / 1000

        # self.episode_position_reward_list.append(position_reward)
        # self.episode_pendulum_velocity_reward_list.append(energy_penalty)
        # self.episode_action_reward_list.append(0.0)

        reward = position_reward + energy_penalty

        reward = max(0.0, reward)

        # print(position_reward, energy_penalty, reward)

        return reward

    def get_reward_for_double_rip_1(self):
        terminal, position_score = self._terminal()

        position_reward = position_score + 4 if terminal else position_score + 3
        if self.is_upright:
            position_reward += 2

        alpha_pendulum_1_velocity = 0.5
        alpha_pendulum_2_velocity = 0.5
        alpha_motor_velocity = 2.0
        energy_penalty_denominator = 500

        energy_penalty = -2.0 * (
            alpha_pendulum_1_velocity * abs(self.pendulum_1_velocity) +
            alpha_pendulum_2_velocity * abs(self.pendulum_2_velocity) +
            alpha_motor_velocity * abs(self.motor_velocity)
        ) / energy_penalty_denominator

        # print(self.pendulum_1_velocity, self.pendulum_2_velocity, self.motor_velocity, energy_penalty)
        # self.episode_position_reward_list.append(position_reward)
        # self.episode_pendulum_velocity_reward_list.append(energy_penalty)
        # self.episode_action_reward_list.append(0.0)

        reward = position_reward + energy_penalty
        # if self.is_upright:
        #     print(
        #         "position_reward: {0:3.4f}".format(position_reward),
        #         "energy_penalty: {0:3.4f}".format(energy_penalty),
        #         "reward : {0:3.4f}".format(reward),
        #         "terminal" if terminal else "",
        #         "upright" if self.is_upright else ""
        #     )

        reward = max(0.0, reward)

        # if self.pendulum_type in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
        #     if reward == 0.0:
        #         self.unit_time = 0.06
        #     else:
        #         self.unit_time = np.clip(0.06 / (sigmoid_2(0.01) * 10), 0.006, 0.06)
        #
        # if self.too_much_rotate and not self.is_upright:
        #     reward = -3.0

        #print(position_reward, energy_penalty, reward)

        return reward

    def get_reward_for_double_rip_2(self):
        #adjusted 1
        if self.pendulum_1_position < 0:
            if (abs(self.pendulum_1_position) % (2.0 * math.pi)) > math.pi:
                #plus
                # (pendulum_1_position % (-2.0 * math.pi)): -2PI ~ -PI
                adjusted_pendulum_1_position = 2.0 * math.pi + (self.pendulum_1_position % (-2.0 * math.pi))
                assert 0 <= adjusted_pendulum_1_position <= math.pi
            else:
                #minus
                adjusted_pendulum_1_position = self.pendulum_1_position % (-2.0 * math.pi)
                assert -math.pi <= adjusted_pendulum_1_position <= 0
        else:
            if (abs(self.pendulum_1_position) % (2.0 * math.pi)) > math.pi:
                #minus
                adjusted_pendulum_1_position = -2.0 * math.pi + (self.pendulum_1_position % (2.0 * math.pi))
                assert -math.pi <= adjusted_pendulum_1_position <= 0
            else:
                #plus
                adjusted_pendulum_1_position = self.pendulum_1_position % (2.0 * math.pi)
                assert 0 <= adjusted_pendulum_1_position <= math.pi
        # print("===============================================")
        # print("Adjusted 1 : ", adjusted_pendulum_1_position)
        #adjusted 2
        if self.pendulum_2_position < 0:
            if (abs(self.pendulum_2_position) % (2.0 * math.pi)) > math.pi:
                # plus
                adjusted_pendulum_2_position = 2.0 * math.pi + (self.pendulum_2_position % (-2.0 * math.pi))
            else:
                # minus
                adjusted_pendulum_2_position = self.pendulum_2_position % (-2.0 * math.pi)
        else:
            if (abs(self.pendulum_2_position) % (2.0 * math.pi)) > math.pi:
                # minus
                adjusted_pendulum_2_position = -2.0 * math.pi + (self.pendulum_2_position % (2.0 * math.pi))
            else:
                # plus
                adjusted_pendulum_2_position = self.pendulum_2_position % (2.0 * math.pi)
        # print("adjusted 2 : ", adjusted_pendulum_2_position)
        #create reward 2 pendulum
        if self.pendulum_1_position < 0:
            if abs(self.pendulum_1_position % (2.0 * math.pi)) > math.pi:
                if self.pendulum_2_position < 0:
                    if abs(self.pendulum_2_position % (2.0 * math.pi)) > math.pi:
                        reward_pendulum_2 = abs(adjusted_pendulum_1_position + adjusted_pendulum_2_position)
                        #plus
                    else:
                        reward_pendulum_2 = abs(adjusted_pendulum_2_position + adjusted_pendulum_1_position)
                        #minus
                else:
                    if abs(self.pendulum_2_position % (2.0 * math.pi)) > math.pi:
                        reward_pendulum_2 = abs(adjusted_pendulum_2_position + adjusted_pendulum_1_position)
                        #minus
                    else:
                        reward_pendulum_2 = abs(adjusted_pendulum_1_position + adjusted_pendulum_2_position)
                        #plus
                # plus
            else:
                if self.pendulum_2_position < 0:
                    if abs(self.pendulum_2_position % (2.0 * math.pi)) > math.pi:
                        reward_pendulum_2 = abs(adjusted_pendulum_2_position + adjusted_pendulum_1_position)
                    else:
                        reward_pendulum_2 = abs(adjusted_pendulum_1_position + adjusted_pendulum_2_position)
                else:
                    if abs(self.pendulum_2_position % (2.0 * math.pi)) > math.pi:
                        reward_pendulum_2 = abs(adjusted_pendulum_1_position + adjusted_pendulum_2_position)
                    else:
                        reward_pendulum_2 = abs(adjusted_pendulum_2_position + adjusted_pendulum_1_position)
                # minus
        else:
            if abs(self.pendulum_1_position % (2.0 * math.pi)) > math.pi:
                if self.pendulum_2_position < 0:
                    if abs(self.pendulum_2_position % (2.0 * math.pi)) > math.pi:
                        reward_pendulum_2 = abs(adjusted_pendulum_2_position + adjusted_pendulum_1_position)
                    else:
                        reward_pendulum_2 = abs(adjusted_pendulum_1_position + adjusted_pendulum_2_position)
                else:
                    if abs(self.pendulum_2_position % (2.0 * math.pi)) > math.pi:
                        reward_pendulum_2 = abs(adjusted_pendulum_1_position + adjusted_pendulum_2_position)
                    else:
                        reward_pendulum_2 = abs(adjusted_pendulum_2_position + adjusted_pendulum_1_position)
                # minus
            else:
                if self.pendulum_2_position < 0:
                    if abs(self.pendulum_2_position % (2.0 * math.pi)) > math.pi:
                        reward_pendulum_2 = abs(adjusted_pendulum_1_position + adjusted_pendulum_2_position)
                        # plus

                    else:
                        reward_pendulum_2 = abs(adjusted_pendulum_2_position + adjusted_pendulum_1_position)
                        # minus
                else:
                    if abs(self.pendulum_2_position % (2.0 * math.pi)) > math.pi:
                        reward_pendulum_2 = abs(adjusted_pendulum_2_position + adjusted_pendulum_1_position)
                        # minus
                    else:
                        reward_pendulum_2 = abs(adjusted_pendulum_1_position + adjusted_pendulum_2_position)

        if reward_pendulum_2 > math.pi:
            reward_pendulum_2 = (2.0 * math.pi) - reward_pendulum_2

        # print("reward_pendulum_2 :", reward_pendulum_2)

        # if abs(adjusted_pendulum_1_position) < math.pi * 0.5:
        #     position_reward = 0
        # else:
        #     if abs(reward_pendulum_2) > abs(adjusted_pendulum_1_position):
        #         position_reward = abs(reward_pendulum_2) + abs(adjusted_pendulum_1_position)
        #     else:
        #         position_reward = 0

        position_reward = abs(reward_pendulum_2) + abs(adjusted_pendulum_1_position)
        alpha_pendulum_1_velocity = 0.3
        alpha_pendulum_2_velocity = 0.3
        alpha_motor_velocity = 0.5
        energy_penalty_denominator = 150

        energy_penalty = -1.0 * (
                alpha_pendulum_1_velocity * abs(self.pendulum_1_velocity) +
                alpha_pendulum_2_velocity * abs(self.pendulum_2_velocity) +
                alpha_motor_velocity * abs(self.motor_velocity)
        ) / energy_penalty_denominator

        # self.episode_position_reward_list.append(position_reward)
        # self.episode_pendulum_velocity_reward_list.append(energy_penalty)
        # self.episode_action_reward_list.append(0.0)

        reward = position_reward + energy_penalty
        # print(
        #     "position_reward: {0:3.4f}".format(position_reward),
        #     "position_Reward_2 : {0:3.4f}".format(reward_pendulum_2),
        #     "energy_penalty: {0:3.4f}".format(energy_penalty),
        #     "reward : {0:3.4f}".format(reward)
        # )
        reward = max(0.0, reward)
        # time.sleep(0.5)
        return reward

    def get_reward_for_double_rip_3(self):
        terminal, position_score = self._terminal()
        position_reward = 0. if not terminal else position_score
        # position_reward = position_score

        alpha_pendulum_1_velocity = 0.3
        alpha_pendulum_2_velocity = 0.3
        alpha_motor_velocity = 0.5
        energy_penalty_denominator = 150

        energy_penalty = -1.0 * (
                alpha_pendulum_1_velocity * abs(self.pendulum_1_velocity) +
                alpha_pendulum_2_velocity * abs(self.pendulum_2_velocity) +
                alpha_motor_velocity * abs(self.motor_velocity)
        ) / energy_penalty_denominator

        # self.episode_position_reward_list.append(position_reward)
        # self.episode_pendulum_velocity_reward_list.append(energy_penalty)
        # self.episode_action_reward_list.append(0.0)

        reward = position_reward + energy_penalty
        # print(
        #     "position_reward: {0:3.4f}".format(position_reward),
        #     "energy_penalty: {0:3.4f}".format(energy_penalty),
        #     "reward : {0:3.4f}".format(reward)
        # )
        reward = max(0.0, reward)

        return reward

    def get_reward_for_double_rip_4(self, pendulum_1_position, pendulum_2_position):
        if pendulum_1_position > 0.0:
            pendulum_1_position = pendulum_1_position % (2.0 * math.pi)
            if pendulum_1_position > math.pi:
                adjusted_pendulum_1_position = pendulum_1_position - (2.0 * math.pi)
                best_pendulum_2_position = math.pi + adjusted_pendulum_1_position
            else:
                adjusted_pendulum_1_position = pendulum_1_position
                best_pendulum_2_position = adjusted_pendulum_1_position - math.pi
        else:
            pendulum_1_position = -pendulum_1_position
            pendulum_1_position = pendulum_1_position % (2.0 * math.pi)
            pendulum_1_position = -pendulum_1_position
            if pendulum_1_position < -math.pi:
                adjusted_pendulum_1_position = pendulum_1_position + (2.0 * math.pi)
                best_pendulum_2_position = adjusted_pendulum_1_position - math.pi
            else:
                adjusted_pendulum_1_position = pendulum_1_position
                best_pendulum_2_position = math.pi + adjusted_pendulum_1_position

        if pendulum_2_position > 0.0:
            pendulum_2_position = pendulum_2_position % (2.0 * math.pi)
            if pendulum_2_position > math.pi:
                adjusted_pendulum_2_position = pendulum_2_position - (2 * math.pi)
            else:
                adjusted_pendulum_2_position = pendulum_2_position
        else:
            pendulum_2_position = -pendulum_2_position
            pendulum_2_position = pendulum_2_position % (2.0 * math.pi)
            pendulum_2_position = -pendulum_2_position
            if pendulum_2_position < -math.pi:
                adjusted_pendulum_2_position = pendulum_2_position + (2.0 * math.pi)
            else:
                adjusted_pendulum_2_position = pendulum_2_position

        reward_pendulum_2 = math.pi - abs(best_pendulum_2_position - adjusted_pendulum_2_position)
        position_reward = abs(reward_pendulum_2) + abs(adjusted_pendulum_1_position)

        energy_penalty = -1.0 * (abs(self.pendulum_1_velocity) + abs(self.pendulum_2_velocity) + 1.5 * abs(self.motor_velocity)) / 150

        # self.episode_position_reward_list.append(position_reward)
        # self.episode_pendulum_velocity_reward_list.append(energy_penalty)
        # self.episode_action_reward_list.append(0.0)

        reward = position_reward + energy_penalty

        reward = max(0.0, reward)

        return reward

    def _terminal(self):
        # ns[0] = wrap(ns[0], -pi, pi)
        # ns[1] = wrap(ns[1], -pi, pi)
        pendulum_1_position = wrap(self.pendulum_1_position, -math.pi, math.pi)
        pendulum_2_position = wrap(self.pendulum_2_position, -math.pi, math.pi)
        is_terminal = bool(-math.cos(pendulum_1_position) - math.cos(pendulum_2_position + pendulum_1_position) > 1.)
        position_score = -2.0 * math.cos(pendulum_1_position) - math.cos(pendulum_2_position + pendulum_1_position)
        # print(
        #     "{0:2.4f}".format(math.degrees(pendulum_1_position)),
        #     "{0:2.4f}".format(math.degrees(pendulum_2_position)),
        #     "is_terminal", is_terminal,
        #     "{0:2.4f}".format(position_score)
        # )

        return is_terminal, position_score

    def render(self, mode='human'):
        pass

    @staticmethod
    def convert_radian_to_degree(radian):
        degree = radian * 180 / math.pi
        return degree

    def stop(self):
        rip_response = self.server_obj.terminate(RipRequest(value=None))

        if rip_response.message != "OK":
            raise ValueError()
