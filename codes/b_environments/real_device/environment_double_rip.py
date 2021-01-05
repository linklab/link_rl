import time
import math
import numpy as np
from enum import Enum
import json
import paho.mqtt.client as paho

# MQTT Topic for RIP
import gym

MQTT_SERVER = '192.168.0.10'
MQTT_PUB_TO_DRIP = 'motor_power'
MQTT_PUB_RESET = 'reset'
MQTT_SUB_RESET_COMPLETE = 'reset_complete'
MQTT_SUB_FROM_DRIP = 'next_state'
MQTT_ERROR = 'error'

STATE_SIZE = 6

PUB_ID = 0


class Status(Enum):
    SWING_UP = -1.0
    SWING_UP_TO_BALANCING = 0.5
    BALANCING = 1.0
    BALANCING_TO_SWING_UP = -0.5


class EnvironmentDoubleRIP():
    def __init__(self, owner, action_min, action_max, env_reset=True, params=None):
        self.episode_steps = 0
        self.total_steps = 0
        self.env_reset = env_reset
        self.params = params

        self.pendulum_position = 0
        self.pendulum_velocity = 0
        self.motor_position = 0
        self.motor_velocity = 0

        self.obs_degree = [None, None]
        self.next_obs_degree = [None, None]
        self.simulation_time = 0.0

        self.too_much_rotate = False
        # self.done_torque_threshold = 0.75

        self.max_velocity = 100.0

        self.action_space = gym.spaces.Box(
            low=action_min, high=action_max, shape=(1,),
            dtype=np.float32
        )

        # high = np.array([1., 1., self.max_velocity, 1., 1., action_max, 1.0], dtype=np.float32)
        high = np.array([1., 1., self.max_velocity, 1., 1., 1.], dtype=np.float32)
        low = high * -1.0
        self.observation_space = gym.spaces.Box(
            low=low, high=high,
            dtype=np.float32
        )

        self.n_states = self.observation_space.shape[0]
        self.n_actions = self.action_space.shape[0]

        self.current_status = None

        self.count_continuous_uprights = 0
        self.is_upright = False
        self.initial_motor_position = 0.0

        self.count_swing_up_states = 0
        self.count_balancing_states = 0

        self.count_continuous_swing_up_states = 0
        self.count_continuous_balancing_states = 0

        self.episode_position_reward_list = []
        self.episode_pendulum_velocity_reward_list = []
        self.episode_action_reward_list = []

        self.reset_complete = False
        self.action_complete = False
        self.sample_time = 10/1000

        if owner == "actual_worker":
            self.mqtt_client = paho.Client(client_id="Environment")
            self.mqtt_client.connect(params.MQTT_SERVER, 1883)
            self.mqtt_client.on_connect = self.__on_connect
            self.mqtt_client.on_message = self.__on_message
            # self.client.on_log = __on_log
            self.mqtt_client.loop_start()

    def __on_connect(self, client, userdata, flags, rc):
        print("mqtt broker connected with result code " + str(rc), flush=False)
        self.mqtt_client.subscribe(topic=params.MQTT_SUB_FROM_DRIP)
        self.mqtt_client.subscribe(topic=params.MQTT_SUB_RESET_COMPLETE)

    @staticmethod
    def __on_log(userdata, level, buf):
        print(buf)

    def __on_message(self, client, userdata, msg):
        if msg.topic == params.MQTT_SUB_RESET_COMPLETE:
            print("\n==================== Reset Complete ====================")
            self.reset_complete = True
            servo_info = msg.payload.decode("utf-8").split('|')
            self.motor_position = float(servo_info[0])
            self.motor_velocity = float(servo_info[1])
            self.pendulum_position = float(servo_info[2])
            self.pendulum_velocity = float(servo_info[3])
        elif msg.topic == params.MQTT_SUB_FROM_DRIP:
            print("\n==================== Receive Next States ====================")
            self.action_complete = True
            servo_info = msg.payload.decode("utf-8").split('|')
            self.motor_position = float(servo_info[0])
            self.motor_velocity = float(servo_info[1])
            self.pendulum_position = float(servo_info[2])
            self.pendulum_velocity = float(servo_info[3])

            print(self.motor_position, self.motor_velocity, self.pendulum_position, self.pendulum_velocity)
        else:
            raise ValueError

    def reset(self):
        self.episode_steps = 0
        self.count_swing_up_states = 0
        self.count_balancing_states = 0
        self.count_continuous_swing_up_states = 0
        self.count_continuous_balancing_states = 0
        self.episode_position_reward_list.clear()
        self.episode_pendulum_velocity_reward_list.clear()
#        self.episode_action_reward_list.clear()

        # Pub reset message
        # ================================================================================== #
        self.mqtt_client.publish(topic=MQTT_PUB_RESET, payload="")
        while not self.reset_complete:
            time.sleep(0.001)
        self.reset_complete = False
        # ================================================================================== #

        self.update_current_state(adjusted_radian=0.0)

        state = (
            math.cos(self.pendulum_position),
            math.sin(self.pendulum_position),
            self.pendulum_velocity,
            math.cos(0.0),  # 1.0
            math.sin(0.0),  # 0.0
            self.motor_velocity,
            # self.current_status.value
        )

        # print("q: {0:7.4}, w: {1:7.4f}, time: {2} -- RESET".format(
        #     self.pendulum_position, self.pendulum_velocity, self.simulation_time
        # ))

        self.too_much_rotate = False

        self.count_continuous_uprights = 0
        self.is_upright = False
        self.initial_motor_position = self.motor_position

        return state

    def step(self, action):
        if type(action) is np.ndarray:
            action = action[0]
        action = int(action)

        # ============================================================================================= #
        self.mqtt_client.publish(topic=MQTT_PUB_TO_DRIP, payload="{0}".format(action))

        previous_time = time.perf_counter()

        while not self.action_complete:
            time.sleep(0.001)
        self.action_complete = False

        current_time = time.perf_counter()
        print("time : ", previous_time - current_time)
        while not current_time - previous_time >= self.sample_time:
            time.sleep(0.0001)
            current_time = time.perf_counter()
        # ============================================================================================= #

        self.episode_steps += 1
        self.total_steps += 1

        if action > 0:
            self.num_continuous_positive_torque += 1
        else:
            self.num_continuous_positive_torque = 0

        if action < 0:
            self.num_continuous_negative_torque += 1
        else:
            self.num_continuous_negative_torque = 0

        # print(self.motor_position, math.cos(self.motor_position), math.sin(self.motor_position))

        if abs(self.initial_motor_position - self.motor_position) > 360:
            self.too_much_rotate = True

        done_conditions = [
            self.episode_steps >= 500,
            self.too_much_rotate
            # self.num_continuous_positive_torque >= 30,
            # self.num_continuous_negative_torque >= 30
        ]

        adjusted_radian = self.pendulum_position_to_adjusted_radian()

        # print("action: {0}, q: {1:7.4}, w: {2:7.4f}, adjusted_radian: {3:7.4f}, reward: {4:10.4f}, time: {5}".format(
        #     action, self.pendulum_position, self.pendulum_velocity, adjusted_radian, reward, self.simulation_time
        # ))

        self.update_current_state(adjusted_radian)

        if any(done_conditions):
            done = True
            if params.CH:
                reward = self.CH_ordinary_reward(
                    adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
                )
            else:
                reward = self.get_reward(adjusted_radian)

            info = {
                "count_balancing_states": self.count_balancing_states,
                "count_swing_up_states": self.count_swing_up_states,
                "episode_position_reward_list": sum(self.episode_position_reward_list),
                "episode_pendulum_velocity_reward": sum(self.episode_pendulum_velocity_reward_list),
                "episode_action_reward": sum(self.episode_action_reward_list)
            }

        else:
            done = False
            if params.CH:
                reward = self.CH_ordinary_reward(
                    adjusted_radian, action, self.num_continuous_positive_torque, self.num_continuous_negative_torque
                )
            else:
                reward = self.get_reward(adjusted_radian)

            info = {}

        state = (
            math.cos(self.pendulum_position),
            math.sin(self.pendulum_position),
            self.pendulum_velocity,
            math.cos(self.initial_motor_position - self.motor_position),
            math.sin(self.initial_motor_position - self.motor_position),
            self.motor_velocity,
            # self.current_status.value
        )
        return state, reward, done, info

    def stop(self):
        self.mqtt_client.publish(topic=MQTT_ERROR, payload="")

    def get_n_states(self):
        n_states = self.observation_space.shape[0]
        return n_states

    def get_n_actions(self):
        n_actions = self.action_space.shape[0]
        return n_actions

    @property
    def action_meanings(self):
        action_meanings = ["Joint effort", ]
        return action_meanings

    def pendulum_position_to_adjusted_radian(self):
        # angle을 0과 360 사이 값(양수)으로 조정
        if abs(self.pendulum_position) > 360:
            q_ = abs(self.pendulum_position) % (360)
        else:
            q_ = abs(self.pendulum_position)

        # radian을 0과 math.pi 사이 값(양수)으로 조정: 3 * math.pi / 2 -->  2 * math.pi - 3 * math.pi / 2 --> math.pi / 2
        if q_ > 180:
            adjusted_radian = 360 - q_
        else:
            adjusted_radian = q_

        return adjusted_radian

    def update_current_state(self, adjusted_radian):
        if self.params.CH:
            if math.pi - math.radians(3) < adjusted_radian <= math.pi:
                self.count_continuous_uprights += 1
            else:
                self.count_continuous_uprights = 0
        else:
            if math.pi - math.radians(12) < adjusted_radian <= math.pi:
                self.count_continuous_uprights += 1
            else:
                self.count_continuous_uprights = 0

        if self.count_continuous_uprights >= 1:
            self.is_upright = True
        else:
            self.is_upright = False

        if self.current_status is None:  # RESET
            self.current_status = Status.SWING_UP
            self.count_swing_up_states += 1
            self.count_continuous_swing_up_states += 1
        elif self.current_status == Status.SWING_UP:
            if self.is_upright:  # SWING_UP --> SWING_UP_TO_BALANCING
                self.current_status = Status.SWING_UP_TO_BALANCING
                self.count_continuous_swing_up_states = 0
                self.count_continuous_balancing_states += 1
                self.count_balancing_states += 1
            else:  # SWING_UP --> SWING_UP
                self.current_status = Status.SWING_UP
                self.count_continuous_swing_up_states += 1
                self.count_swing_up_states += 1
        elif self.current_status == Status.SWING_UP_TO_BALANCING:
            if self.is_upright:  # SWING_UP_TO_BALANCING --> BALANCING
                self.current_status = Status.BALANCING
                self.count_continuous_balancing_states += 1
                self.count_balancing_states += 1
            else:  # SWING_UP_TO_BALANCING --> BALANCING_TO_SWING_UP
                self.current_status = Status.BALANCING_TO_SWING_UP
                self.count_swing_up_states += 1
                self.count_continuous_balancing_states = 0
                self.count_continuous_swing_up_states += 1
        elif self.current_status == Status.BALANCING:
            if self.is_upright:  # BALANCING --> BALANCING
                self.current_status = Status.BALANCING
                self.count_balancing_states += 1
                self.count_continuous_balancing_states += 1
            else:  # BALANCING --> BALANCING_TO_SWING_UP
                self.current_status = Status.BALANCING_TO_SWING_UP
                self.count_swing_up_states += 1
                self.count_continuous_balancing_states = 0
                self.count_continuous_swing_up_states += 1
        elif self.current_status == Status.BALANCING_TO_SWING_UP:
            if self.is_upright:  # BALANCING_TO_SWING_UP --> SWING_UP_TI_BALANCING
                self.current_status = Status.SWING_UP_TO_BALANCING
                self.count_balancing_states += 1
                self.count_continuous_swing_up_states = 0
                self.count_continuous_balancing_states += 1
            else:  # BALANCING_TO_SWING_UP --> SWING_UP
                self.current_status = Status.SWING_UP
                self.count_swing_up_states += 1
                self.count_continuous_swing_up_states += 1
        else:
            raise ValueError()

    @staticmethod
    def convert_radian_to_degree(radian):
        degree = radian * 180 / math.pi
        return degree


    def get_reward(self, adjusted_radian):
        if self.too_much_rotate:
            position_reward = -1.0
        else:
            if self.current_status in [Status.SWING_UP]:
                position_reward = 0.0
            elif self.current_status in [Status.SWING_UP_TO_BALANCING]:
                position_reward = 1.0
            else:
                position_reward = adjusted_radian  # math.pi - math.radians(12) ~ math.pi

        self.episode_position_reward_list.append(position_reward)
        self.episode_pendulum_velocity_reward_list.append(0.0)
        self.episode_action_reward_list.append(0.0)

        reward = position_reward

        return reward

    # def CH_ordinary_reward(self, adjusted_radian, action, num_continuous_positive_torque,
    #                        num_continuous_negative_torque):
    #     # reward = -((math.pi - adjusted_radian) ** 2 + 0.1 * (self.pendulum_velocity ** 2) + 0.001 * (action ** 2))
    #     if adjusted_radian < math.pi / 2:
    #         reward = 0.0 - abs(np.tanh(self.motor_velocity)) * 0.1
    #     else:
    #         reward = adjusted_radian - abs(np.tanh(self.motor_velocity)) * 0.1
    #
    #     reward -= num_continuous_positive_torque * 0.01
    #     reward -= num_continuous_negative_torque * 0.01
    #
    #     return reward

    def render(self, mode='human'):
        pass
