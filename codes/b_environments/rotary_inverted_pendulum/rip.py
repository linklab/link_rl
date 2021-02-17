import math
import random

import grpc
import gym
import numpy as np
import sys,os

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.b_environments.rotary_inverted_pendulum import rip_service_pb2_grpc
from codes.b_environments.rotary_inverted_pendulum.rip_service_pb2 import RipRequest
from codes.b_environments.rotary_inverted_pendulum.matlabcode import SimulinkPlant

from codes.e_utils.names import RLAlgorithmName, EnvironmentName

np.set_printoptions(formatter={'float_kind': lambda x: '{0:0.6f}'.format(x)})

BLOWING_ACTION_RATE = 0.0002  # 5000 스텝에 1번 정도(지수 분포)의 주가로 외력이 가해짐 --> Stochastic Env.

RIP_SERVER = '192.168.0.254'

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

        self.max_velocity = 100.0

        # self.action_index_to_voltage = [-0.05, -0.025, -0.008, 0, 0.008, 0.025, 0.05]

        if self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_V0:
            self.action_index_to_voltage = [
                -0.08, -0.05, -0.025, -0.0125, -0.008, -0.002, 0.0, 0.002, 0.008, 0.0125, 0.025, 0.05, 0.08
            ]
        elif self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
            self.action_index_to_voltage = [
                -3.5, -2.75, -2.0, -1.5, -0.75, -0.35, -0.10, -0.05, -0.025, -0.016, 0.0,
                0.016, 0.025, 0.05, 0.10, 0.35, 0.75, 1.5, 2.0, 2.75, 3.5
            ]
        elif self.pendulum_type in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            self.action_index_to_voltage = None
        else:
            raise ValueError()

        if self.params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
            self.action_space = gym.spaces.Discrete(len(self.action_index_to_voltage))
            self.n_actions = self.action_space.n
        else:
            self.action_space = gym.spaces.Box(
                low=self.action_min, high=self.action_max, shape=(1,),
                dtype=np.float32
            )
            self.n_actions = self.action_space.shape[0]

        #high = np.array([1., 1., self.max_velocity, 1., 1., action_max, 1.0], dtype=np.float32)

        if self.pendulum_type in [
            EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.REAL_DEVICE_RIP
        ]:
            high = np.array(
                [1., 1., self.max_velocity, 1., 1., self.max_velocity],
                dtype=np.float32
            )
        elif self.pendulum_type in [
            EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP
        ]:
            high = np.array(
                [1., 1., self.max_velocity, 1., 1., self.max_velocity, 1., 1., self.max_velocity],
                dtype=np.float32
            )
        else:
            raise ValueError()

        low = high * -1.0

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        self.n_states = self.observation_space.shape[0]

        self.current_status = None

        self.count_continuous_uprights = 0
        self.is_upright = False
        self.initial_motor_position = 0.0

        self.episode_position_reward_list = []
        self.episode_pendulum_velocity_reward_list = []
        self.episode_action_reward_list = []

        self.next_time_step_of_external_blow = int(random.expovariate(BLOWING_ACTION_RATE))

        self.num_episodes = 0
        self.episode_period_env_reset_forced = 10

        if self.pendulum_type in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            channel = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER))
            self.server_obj = rip_service_pb2_grpc.RDIPStub(channel)
        else:
            self.server_obj = None

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

        if self.total_steps == 0:
            print("next_time_step_of_external_blow: {0}".format(
                self.next_time_step_of_external_blow
            ))

        if self.env_reset or self.num_episodes % self.episode_period_env_reset_forced == 0:
            print("ENV RESET")
            self.plant.connectStart()

        self.episode_position_reward_list.clear()
        self.episode_pendulum_velocity_reward_list.clear()
        self.episode_action_reward_list.clear()

        if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.REAL_DEVICE_RIP]:


            self.update_current_state(adjusted_pendulum_1_radian=0.0)
        elif self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
            if self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
                self.pendulum_1_position, self.motor_position, self.pendulum_2_position, self.pendulum_1_velocity, \
                self.motor_velocity, self.pendulum_2_velocity, self.simulation_time = self.plant.getHistory()
            else:
                rip_response = self.server_obj.reset(RipRequest(value=None))

                self.motor_position = rip_response.arm_angle
                self.motor_velocity = rip_response.arm_velocity
                self.pendulum_1_position = rip_response.link_1_angle
                self.pendulum_1_velocity = rip_response.link_1_velocity
                self.pendulum_2_position = rip_response.link_2_angle
                self.pendulum_2_velocity = rip_response.link_2_velocity
                self.simulation_time = None

            state = (
                math.cos(self.pendulum_1_position),
                math.sin(self.pendulum_1_position),
                self.pendulum_1_velocity,
                math.cos(self.pendulum_2_position),
                math.sin(self.pendulum_2_position),
                self.pendulum_2_velocity,
                math.cos(0.0),  # 1.0
                math.sin(0.0),  # 0.0
                self.motor_velocity,
            )

            self.update_current_state_for_double_rip(adjusted_pendulum_1_radian=0.0, adjusted_pendulum_2_radian=0.0)
        else:
            raise ValueError()

        self.too_much_rotate = False

        self.count_continuous_uprights = 0
        self.is_upright = False

        self.initial_motor_position = self.motor_position

        return state

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
        self.episode_steps += 1

        self.total_steps += 1
        if self.total_steps >= self.next_time_step_of_external_blow:
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
                action = random.uniform(
                    a=self.action_index_to_voltage[0] * 10.0,
                    b=self.action_index_to_voltage[-1] * 10.0,
                )
            elif self.params.RL_ALGORITHM in [
                RLAlgorithmName.DDPG_V0,
                RLAlgorithmName.CONTINUOUS_A2C_V0,
                RLAlgorithmName.CONTINUOUS_PPO_V0
            ]:
                action = random.uniform(
                    a=self.action_min * 10.0,
                    b=self.action_max * 10.0
                )
            else:
                raise ValueError()

            self.next_time_step_of_external_blow = self.total_steps + int(random.expovariate(BLOWING_ACTION_RATE))
            print("External Blow: {0:7.5f}, next_time_step_of_external_blow: {1}".format(
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
        elif self.pendulum_type == EnvironmentName.REAL_DEVICE_RIP:
            rip_response = self.server_obj.step(RipRequest(value=action))
            self.motor_position = rip_response.arm_angle
            self.motor_velocity = rip_response.arm_velocity
            self.pendulum_1_position = rip_response.link_1_angle
            self.pendulum_1_velocity = rip_response.link_1_velocity
            self.simulation_time = None
        elif self.pendulum_type == EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0:
            self.plant.simulate(action)

            self.pendulum_1_position, self.motor_position, self.pendulum_2_position, self.pendulum_1_velocity, \
            self.motor_velocity, self.pendulum_2_velocity, self.simulation_time = self.plant.getHistory()
        elif self.pendulum_type == EnvironmentName.REAL_DEVICE_DOUBLE_RIP:
            rip_response = self.server_obj.step(RipRequest(value=action))

            self.motor_position = rip_response.arm_angle
            self.motor_velocity = rip_response.arm_velocity
            self.pendulum_1_position = rip_response.link_1_angle
            self.pendulum_1_velocity = rip_response.link_1_velocity
            self.pendulum_2_position = rip_response.link_2_angle
            self.pendulum_2_velocity = rip_response.link_2_velocity
            self.simulation_time = None
        else:
            raise ValueError()

        #print(self.motor_position, math.cos(self.motor_position), math.sin(self.motor_position))

        if abs(self.initial_motor_position - self.motor_position) > math.pi * 2:
            self.too_much_rotate = True

        done_conditions = [
            self.episode_steps >= 10000,
            self.episode_steps >= 500 and not self.is_upright,
            self.too_much_rotate and not self.is_upright
        ]

        adjusted_pendulum_1_radian = self.pendulum_position_to_adjusted_radian(self.pendulum_1_position)
        adjusted_pendulum_2_radian = self.pendulum_position_to_adjusted_radian(self.pendulum_2_position)

        # print("action: {0}, q: {1:7.4}, w: {2:7.4f}, adjusted_radian: {3:7.4f}, reward: {4:10.4f}, time: {5}".format(
        #     action, self.pendulum_1_position, self.pendulum_1_velocity, adjusted_radian, reward, self.simulation_time
        # ))

        if self.pendulum_type in [
            EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.REAL_DEVICE_RIP
        ]:
            self.update_current_state(adjusted_pendulum_1_radian)
            reward = self.get_reward(adjusted_pendulum_1_radian)
        elif self.pendulum_type in [
            EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP
        ]:
            self.update_current_state_for_double_rip(adjusted_pendulum_1_radian, adjusted_pendulum_2_radian)
            reward = self.get_reward_for_double_rip(adjusted_pendulum_1_radian, adjusted_pendulum_2_radian)
        else:
            raise ValueError()

        if any(done_conditions):
            done = True

            info = {
                "adjusted_pendulum_1_radian": adjusted_pendulum_1_radian,
                "adjusted_pendulum_2_radian": adjusted_pendulum_2_radian if self.pendulum_type in [EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0, EnvironmentName.REAL_DEVICE_DOUBLE_RIP] else None,
                "episode_position_reward_list": sum(self.episode_position_reward_list),
                "episode_pendulum_velocity_reward": sum(self.episode_pendulum_velocity_reward_list),
                "episode_action_reward": sum(self.episode_action_reward_list)
            }

            self.num_episodes += 1

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
            state = (
                math.cos(self.pendulum_1_position),
                math.sin(self.pendulum_1_position),
                self.pendulum_1_velocity,
                math.cos(self.pendulum_2_position),
                math.sin(self.pendulum_2_position),
                self.pendulum_2_velocity,
                math.cos(self.initial_motor_position - self.motor_position),
                math.sin(self.initial_motor_position - self.motor_position),
                self.motor_velocity,
            )
        else:
            raise ValueError()
        return state, reward, done, info

    def get_reward(self, adjusted_pendulum_1_radian):
        if self.is_upright:
            position_reward = adjusted_pendulum_1_radian / math.pi  # math.pi - math.radians(12) ~ math.pi
        else:
            position_reward = adjusted_pendulum_1_radian / (math.pi * 2.0)

        energy_penalty = 2.0 * -1.0 * (abs(self.pendulum_1_velocity) + abs(self.motor_velocity)) / 100

        self.episode_position_reward_list.append(position_reward)
        self.episode_pendulum_velocity_reward_list.append(energy_penalty)
        self.episode_action_reward_list.append(0.0)

        reward = position_reward + energy_penalty

        reward = max(0.0, reward)

        # print(position_reward, energy_penalty, reward)

        return reward

    # def get_reward_for_double_rip(self, adjusted_pendulum_1_radian, adjusted_pendulum_2_radian):
    #     combined_radian = adjusted_pendulum_1_radian + (math.pi - adjusted_pendulum_2_radian) / 2.0
    #
    #     if self.is_upright:
    #         position_reward = combined_radian / math.pi  # math.pi - math.radians(12) ~ math.pi
    #     else:
    #         position_reward = combined_radian / math.pi * 2.0
    #
    #     energy_penalty = -1.0 * (abs(self.pendulum_1_velocity) + abs(self.pendulum_2_velocity) + abs(self.motor_velocity)) / 150
    #
    #     self.episode_position_reward_list.append(position_reward)
    #     self.episode_pendulum_velocity_reward_list.append(energy_penalty)
    #     self.episode_action_reward_list.append(0.0)
    #     reward = position_reward + energy_penalty
    #     reward = max(0.0, reward)
    #     # print(position_reward, energy_penalty, reward)
    #
    #     return reward

    def get_reward_for_double_rip(self, pendulum_1_position, pendulum_2_position):
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

        self.episode_position_reward_list.append(position_reward)
        self.episode_pendulum_velocity_reward_list.append(energy_penalty)
        self.episode_action_reward_list.append(0.0)

        reward = position_reward + energy_penalty

        reward = max(0.0, reward)

        return reward

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