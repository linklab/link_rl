import time
import math
import random
import gym
from gym import spaces
import numpy as np
import grpc

# MQTT Topic for RIP
from codes.b_environments.quanser_rotary_inverted_pendulum import quanser_service_pb2_grpc
from codes.e_utils.names import RLAlgorithmName, AgentMode
from common.environments.environment import Environment
from codes.a_config.parameters import PARAMETERS as params
from codes.b_environments.quanser_rotary_inverted_pendulum.quanser_service_pb2 import QuanserRequest

STATE_SIZE = 6

balance_motor_power_list = [-60., 0., 60.]
BLOWING_ACTION_RATE = 0.0004

if params.SERVER_IDX == 3:
    RIP_SERVER = '10.0.0.4'
elif params.SERVER_IDX == 2:
    RIP_SERVER = '10.0.0.5'


def get_quanser_rip_observation_space():
    if params.QUANSER_STATE_INFO == 0:
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1., 1., 500., 1., 1., 500.], dtype=np.float32)
    elif params.QUANSER_STATE_INFO == 1:
        low = np.array([0, 0, 0], dtype=np.float32)
        high = np.array([1., 1., 500.], dtype=np.float32)
    elif params.QUANSER_STATE_INFO == 2:
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([1., 1., 500., 500.], dtype=np.float32)
    else:
        raise ValueError()

    observation_space = gym.spaces.Box(
        low=low, high=high, dtype=np.float32
    )
    n_states = observation_space.shape[0]
    return observation_space, n_states


def get_quanser_rip_action_info(params):
    if params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
        action_index_to_voltage = list(np.array([
            -1.0, -0.75, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0
        ]) * params.ACTION_SCALE)
        action_space = gym.spaces.Discrete(len(action_index_to_voltage))
        num_outputs = 1
        action_n = action_space.n
        action_min = 0
        action_max = action_space.n - 1
    else:
        action_index_to_voltage = None
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,),
            dtype=np.float32
        )
        num_outputs = action_space.shape[0]
        action_n = None
        action_min = action_space.low
        action_max = action_space.high

    return action_space, num_outputs, action_n, action_min, action_max, action_index_to_voltage


class EnvironmentQuanserRIP(gym.Env):
    def __init__(self, mode=AgentMode.TRAIN):
        super(EnvironmentQuanserRIP, self).__init__()
        self.next_time_step_of_external_blow = int(random.expovariate(BLOWING_ACTION_RATE))

        self.episode = 0

        self.params = params

        self.previous_time = time.perf_counter()
        self.state_space_shape = (STATE_SIZE,)
        self.action_space_shape = (len(balance_motor_power_list),)

        self.env_reset = False

        self.reward = 0

        self.step_idx = 0
        self.state = []
        self.episode_steps = 0

        self.count_continuous_uprights = 0
        self.is_upright = False

        self.motor_radian = 0
        self.motor_velocity = 0
        self.pendulum_radian = 0
        self.pendulum_velocity = 0

        self.is_motor_limit = False

        self.unit_time = self.params.UNIT_TIME
        self.over_unit_time = 0

        if mode == AgentMode.PLAY:
            self.max_episode_step = self.params.MAX_EPISODE_STEP_AT_PLAY
        else:
            self.max_episode_step = self.params.MAX_EPISODE_STEP

        self.initial_motor_radian = 0.0
        #==================observation==========================================================
        self.observation_space, self.n_states = get_quanser_rip_observation_space()
        #=======================================================================================

        #==================action===============================================================
        self.action_space, self.num_outputs, self.action_n, self.action_min, self.action_max, \
        self.action_index_to_voltage = get_quanser_rip_action_info(self.params)
        #=======================================================================================

        channel = grpc.insecure_channel('{0}:50051'.format(RIP_SERVER))
        self.server_obj = quanser_service_pb2_grpc.QuanserRIPStub(channel)

        print(self.max_episode_step)
    def get_n_states(self):
        if params.QUANSER_STATE_INFO == 0:
            n_states = 6
        elif params.QUANSER_STATE_INFO == 1:
            n_states = 3
        elif params.QUANSER_STATE_INFO == 2:
            n_states = 4
        else:
            raise ValueError()
        return n_states

    def get_state_shape(self):
        state_shape = (2,)
        return state_shape

    def get_action_shape(self):
        action_shape = (3,)
        return action_shape

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "STOP", "RIGHT"]
        return action_meanings

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "STOP", "RIGHT"]
        return action_meanings

    # def pendulum_reset(self):
    #     quanser_response = self.server_obj.step(QuanserRequest(value=0.0))
    #     if quanser_response.message != "PENDULUM_RESET":
    #         raise ValueError()

    def reset(self):
        self.episode_steps = 0
        self.reward = 0
        self.is_motor_limit = False

        quanser_response = self.server_obj.reset(QuanserRequest(value=0.0))
        if quanser_response.message != "RESET":
            raise ValueError()

        if self.step_idx == 0:
            print("next_time_step_of_external_blow: {0}".format(
                self.next_time_step_of_external_blow
            ))

        self.motor_radian = quanser_response.motor_radian
        self.motor_velocity = quanser_response.motor_velocity
        self.pendulum_radian = quanser_response.pendulum_radian
        self.pendulum_velocity = quanser_response.pendulum_velocity

        # if self.episode % 5 == 0:
        #     print("** [EPISODE: {0}] RESET PENDULUM RADIAN : {1:6.3f}, "
        #           "math.cos(self.pendulum_radian): {2:6.3f}, math.sin(self.pendulum_radian): {3:6.3f}".format(
        #         self.episode,
        #         self.pendulum_radian,
        #         math.cos(self.pendulum_radian),
        #         math.sin(self.pendulum_radian),
        #     ))

        if self.params.QUANSER_STATE_INFO == 0:
            self.state = [
                math.cos(self.pendulum_radian),
                math.sin(self.pendulum_radian),
                self.pendulum_velocity / params.VELOCITY_STATE_DENOMINATOR,
                # math.cos(0.0),
                # math.sin(0.0),
                math.cos(quanser_response.motor_radian),
                math.sin(quanser_response.motor_radian),
                self.motor_velocity / params.VELOCITY_STATE_DENOMINATOR
            ]
        elif self.params.QUANSER_STATE_INFO == 1:
            self.state = [
                math.cos(self.pendulum_radian),
                math.sin(self.pendulum_radian),
                self.pendulum_velocity / params.VELOCITY_STATE_DENOMINATOR,
                # math.cos(0.0),
                # math.sin(0.0),
                # math.cos(quanser_response.motor_radian),
                # math.sin(quanser_response.motor_radian),
                # self.motor_velocity / params.VELOCITY_STATE_DENOMINATOR
            ]
        elif self.params.QUANSER_STATE_INFO == 2:
            self.state = [
                math.cos(self.pendulum_radian),
                math.sin(self.pendulum_radian),
                self.pendulum_velocity / params.VELOCITY_STATE_DENOMINATOR,
                # math.cos(0.0),
                # math.sin(0.0),
                # math.cos(quanser_response.motor_radian),
                # math.sin(quanser_response.motor_radian),
                self.motor_velocity / params.VELOCITY_STATE_DENOMINATOR
            ]
        else:
            raise ValueError()
        # wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3
        # wait_time = 1
        # previousTime = time.perf_counter()
        # time_done = False
        # while not time_done:
        #     currentTime = time.perf_counter()
        #     if currentTime - previousTime >= wait_time:
        #         time_done = True
        #     time.sleep(0.001)

        self.initial_motor_radian = self.motor_radian
        self.is_motor_limit = False
        self.episode += 1

        return np.asarray(self.state)

    def step(self, action):
        # current_time = time.perf_counter()
        # print("current_time - self.previous_time", current_time - self.previous_time)
        while True:
            current_time = time.perf_counter()
            if current_time - self.previous_time >= self.unit_time:
                break
            time.sleep(0.0001)

        current_time = time.perf_counter()
        step_time = current_time - self.previous_time

        if step_time > self.unit_time:
            self.over_unit_time += 1

        # print(self.step_idx, action, step_time)
        self.previous_time = time.perf_counter()

        if self.step_idx % 100000 == 0:
            print("*OVER UNIT TIME STEP NUMBER :", self.over_unit_time)
        #######################################################


        if self.step_idx >= self.next_time_step_of_external_blow:
            if self.params.RL_ALGORITHM in [
                RLAlgorithmName.DQN_V0, RLAlgorithmName.DISCRETE_A2C_V0, RLAlgorithmName.DISCRETE_PPO_V0
            ]:
                action = random.randint(a=0, b=len(self.action_index_to_voltage)-1)
                action = action * self.action_index_to_voltage[action] * 2.0
            elif self.params.RL_ALGORITHM in [
                RLAlgorithmName.DDPG_V0,
                RLAlgorithmName.CONTINUOUS_A2C_V0,
                RLAlgorithmName.CONTINUOUS_PPO_V0,
                RLAlgorithmName.TD3_V0,
                RLAlgorithmName.CONTINUOUS_SAC_V0,
            ]:
                action = random.uniform(a=-1.0, b=1.0)
                action = action * self.params.ACTION_SCALE * 2.0
            else:
                raise ValueError()

            self.next_time_step_of_external_blow = self.step_idx + int(random.expovariate(BLOWING_ACTION_RATE))
            print("[{0:6}/{1}] External Blow: {2:7.5f}, next_time_step_of_external_blow: {3}".format(
                self.step_idx + 1,
                self.params.MAX_GLOBAL_STEP,
                action,
                self.next_time_step_of_external_blow
            ))
        else:
            action = action[0]
            if self.params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
                action = self.action_index_to_voltage[int(action)]
            elif self.params.RL_ALGORITHM in [
                RLAlgorithmName.DDPG_V0,
                RLAlgorithmName.CONTINUOUS_A2C_V0,
                RLAlgorithmName.CONTINUOUS_PPO_V0,
                RLAlgorithmName.TD3_V0,
                RLAlgorithmName.CONTINUOUS_SAC_V0,
            ]:
                action = action * self.params.ACTION_SCALE

        #motor_power = float(action)
        # if self.step_idx % 100 == 0:
        #     print(action)

        #==================== Grpc and use sample time========================================
        # previous_time = time.perf_counter()
        quanser_response = self.server_obj.step(QuanserRequest(value=action))
        # print(motor_power)
        # if quanser_response.message != "STEP":
        #     raise ValueError()
        # while True:
        #     current_time = time.perf_counter()
        #     if current_time - previous_time >= 25 / 1000:
        #         break
        #     time.sleep(0.0001)
        #=====================================================================================

        self.motor_radian = quanser_response.motor_radian
        self.motor_velocity = quanser_response.motor_velocity
        self.pendulum_radian = quanser_response.pendulum_radian
        self.pendulum_velocity = quanser_response.pendulum_velocity
        self.is_motor_limit = quanser_response.is_motor_limit
        # print("motor: ", self.motor_radian)
        # print("===========transfer time : {0:1.6f} action : {1:3.2f}".format(previous_time - time.perf_counter(), action))
        # print("motor radian : {0:1.3f}, motor velocity : {1:1.3f}, pendulum radian : {2:1.3f}, pendulum velocity : {3:1.3f}".format(
        #     math.degrees(self.motor_radian), self.motor_velocity,
        #       self.pendulum_radian, self.pendulum_velocity
        #     ))
        # print("is motor limit", self.is_motor_limit)

        if self.params.QUANSER_STATE_INFO == 0:
            self.state = [
                math.cos(self.pendulum_radian),
                math.sin(self.pendulum_radian),
                self.pendulum_velocity / params.VELOCITY_STATE_DENOMINATOR,
                # math.cos(0.0),
                # math.sin(0.0),
                math.cos(quanser_response.motor_radian),
                math.sin(quanser_response.motor_radian),
                self.motor_velocity / params.VELOCITY_STATE_DENOMINATOR
            ]
        elif self.params.QUANSER_STATE_INFO == 1:
            self.state = [
                math.cos(self.pendulum_radian),
                math.sin(self.pendulum_radian),
                self.pendulum_velocity / params.VELOCITY_STATE_DENOMINATOR,
                # math.cos(0.0),
                # math.sin(0.0),
                # math.cos(quanser_response.motor_radian),
                # math.sin(quanser_response.motor_radian),
                # self.motor_velocity / params.VELOCITY_STATE_DENOMINATOR
            ]
        elif self.params.QUANSER_STATE_INFO == 2:
            self.state = [
                math.cos(self.pendulum_radian),
                math.sin(self.pendulum_radian),
                self.pendulum_velocity / params.VELOCITY_STATE_DENOMINATOR,
                # math.cos(0.0),
                # math.sin(0.0),
                # math.cos(quanser_response.motor_radian),
                # math.sin(quanser_response.motor_radian),
                self.motor_velocity / params.VELOCITY_STATE_DENOMINATOR
            ]
        else:
            raise ValueError()
        next_state = np.asarray(self.state)

        self.update_current_state()

        done, info = self.__isDone()

        #=======================reward================================================================
        self.reward = self.get_reward(action)
        # print(self.reward, self.pendulum_radian)
        #=============================================================================================
        self.step_idx += 1
        self.episode_steps += 1

        # print("pendulum radian : {0}, motor radian: {1}, reward: {2}, pendulum_velocity : {3} \n\n".format(
        #     self.pendulum_radian, self.motor_radian, self.is_motor_limit, self.pendulum_velocity
        # ))

        return next_state, self.reward, done, info

    def __isDone(self):
        info = {}

        def insert_to_info(s):
            info["result"] = s

        if self.episode_steps >= self.max_episode_step: # 5000 * 25ms (0.025sec.) = 125 sec.
            insert_to_info("*** Success ***")
            return True, info
        # elif self.is_motor_limit:
        #     print(self.motor_radian, " !!!!!!!")
        #     insert_to_info("*** Limit position ***")
        #     return True, info
        elif self.is_motor_limit:#abs(self.motor_radian) > math.radians(90) or self.is_motor_limit:
            insert_to_info("***motor_radian exceed 90***")
            return True, info
        else:
            insert_to_info("")
            return False, info

    def update_current_state(self):
        if abs(self.pendulum_radian) < math.radians(90):
            self.count_continuous_uprights += 1
        else:
            self.count_continuous_uprights = 0

        if self.count_continuous_uprights >= 1:
            self.is_upright = True
        else:
            self.is_upright = False

    # def get_reward(self):
    #     if abs(self.pendulum_radian) > math.radians(30):
    #         position_reward = 0.0
    #     else:
    #         if self.is_upright:
    #             position_reward = math.pi - abs(self.pendulum_radian)  # math.pi - math.radians(12) ~ math.pi
    #         else:
    #             if self.is_motor_limit:
    #                 position_reward = 0.0
    #             else:
    #                 position_reward = (math.pi - abs(self.pendulum_radian)) / 2
    #
    #     energy_penalty = 2.0 * -1.0 * (abs(self.pendulum_velocity) + abs(self.motor_velocity)) / 100
    #
    #     reward = position_reward + energy_penalty
    #
    #     reward = max(0.0, reward)
    #
    #     return reward

    def get_reward(self, action):
        # print(self.is_upright, self.pendulum_radian, math.radians(90), action)
        if self.is_upright:
            position_reward = math.pi - abs(self.pendulum_radian)# math.pi - math.radians(12) ~ math.pi
        else:
            if self.is_motor_limit:
                position_reward = 0.0
            else:
                position_reward = (math.pi - abs(self.pendulum_radian)) / 2
                # position_reward = 1 + math.cos(self.pendulum_radian)

        energy_penalty = -1.0 * (abs(self.pendulum_velocity) + abs(self.motor_velocity)) / 100

        reward = position_reward + energy_penalty

        reward = max(0.000001, reward) / params.REWARD_DENOMINATOR

        # open ai pendulum reward = -(theta^2 + 0.1theta_dt^2 + 0.001action^2)
        # reward = -((self.pendulum_radian**2) + 0.1*((self.pendulum_velocity/100)**2) + 0.001*(((2*action)/params.ACTION_SCALE)**2))

        # print(position_reward, energy_penalty, reward)

        return reward

    def render(self):
        pass