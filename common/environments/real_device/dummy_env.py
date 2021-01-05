import time
import math
import numpy as np
from enum import Enum
import json
import paho.mqtt.client as paho
import random

# MQTT Topic for RIP
import gym
from common.environments.environment import Environment
from config.parameters import PARAMETERS as params

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


class EnvironmentDoubleRIP(Environment):
    def __init__(self):
        self.mqtt_client = paho.Client(client_id="Environment")
        self.mqtt_client.connect(params.MQTT_SERVER, 1883)
        self.mqtt_client.on_connect = self.__on_connect
        self.mqtt_client.on_message = self.__on_message
        # self.client.on_log = __on_log
        self.mqtt_client.loop_start()
        self.total_steps = 0

    def __on_connect(self, client, userdata, flags, rc):
        print("mqtt broker connected with result code " + str(rc), flush=False)
        self.mqtt_client.subscribe(topic=params.MQTT_SUB_FROM_DRIP)
        self.mqtt_client.subscribe(topic=params.MQTT_SUB_RESET_COMPLETE)

    @staticmethod
    def __on_log(userdata, level, buf):
        print(buf)

    def __on_message(self, client, userdata, msg):
        if msg.topic == params.MQTT_SUB_RESET_COMPLETE:
            print("@@@ Reset Complete @@@")
            self.reset_complete = True

        elif msg.topic == params.MQTT_SUB_FROM_DRIP:
            servo_info = msg.payload.decode("utf-8").split('|')
            motor_position = float(servo_info[0])
            motor_velocity = float(servo_info[1])
            pendulum_position = float(servo_info[2])
            pendulum_velocity = float(servo_info[3])
            # print(motor_velocity, motor_position)
            self.set_state(motor_position, motor_velocity, pendulum_position, pendulum_velocity)

        else:
            raise ValueError

    def step(self):
        print("@@")
        for i in range(100000):
            try:
                plus_action = random.randrange(300,900)
                minus_action = random.randrange(-900,-300)

                action = random.choice([plus_action, minus_action])
                print(action)
                # ============================================================================================= #
                self.mqtt_client.publish(topic=MQTT_PUB_TO_DRIP, payload="{0}".format(action))
                # ============================================================================================= #

                self.total_steps += 1
                time.sleep(0.005)

            except KeyboardInterrupt:
                print("stop")
                self.stop()
        self.stop()

    def stop(self):
        action = 0
        self.mqtt_client.publish(topic=MQTT_PUB_TO_DRIP, payload="{0}".format(action))

def main():
    test_env = EnvironmentDoubleRIP()
    test_env.step()
    test_env.stop()

if __name__ == "__main__":
    main()


