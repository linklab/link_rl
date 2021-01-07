# -*- coding:utf-8 -*-
import pickle
import time
import traceback
import zlib
import sys, os
import paho.mqtt.client as mqtt
import numpy as np
import torch

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils
from codes.e_utils.logger import get_logger

from codes.f_main.federation_main.chief_workers.chief import Chief

logger = get_logger("chief")

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

try:
    env = rl_utils.get_environment(params=params)
    input_shape = env.observation_space.shape
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    rl_model = rl_utils.get_rl_model(
        worker_id=-1, input_shape=input_shape, num_inputs=num_inputs, num_outputs=num_outputs,
        params=params, device=device
    )
    chief = Chief(logger=logger, rl_model=rl_model, params=params)
except:
    traceback.print_exc()
    sys.exit(-1)


def on_chief_connect(client, userdata, flags, rc):
    msg = "Chief is successfully connected with broker@{0}".format(params.MQTT_SERVER)
    logger.info(msg)
    client.subscribe(params.MQTT_TOPIC_EPISODE_DETAIL)
    client.subscribe(params.MQTT_TOPIC_SUCCESS_DONE)
    client.subscribe(params.MQTT_TOPIC_FAIL_DONE)
    print(msg)


def on_chief_log(mqttc, obj, level, string):
    print(string)


def on_chief_message(client, userdata, msg):
    try:
        msg_payload = zlib.decompress(msg.payload)
        msg_payload = pickle.loads(msg_payload)

        log_msg = "[RECV] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'worker_id': {2}, 'loss': {3:8.4}".format(
            msg.topic,
            msg_payload['episode'],
            msg_payload['worker_id'],
            msg_payload['loss'],
        )

        if 'actor_objective' in msg_payload:
            log_msg += ", 'actor_objective': {0}".format(msg_payload['actor_objective'])

        if 'episode_reward' in msg_payload:
            log_msg += ", 'episode_reward': {0}".format(msg_payload['episode_reward'])

        if 'agent_type' in msg_payload:
            log_msg += ", 'agent_type': {0}".format(msg_payload['agent_type'])

        if params.MODE_PARAMETERS_TRANSFER and msg.topic == params.MQTT_TOPIC_SUCCESS_DONE:
            log_msg += ", 'parameters_length': {0}".format(len(msg_payload['parameters']))
        elif params.MODE_GRADIENTS_UPDATE and msg.topic == params.MQTT_TOPIC_EPISODE_DETAIL:
            log_msg += ", 'gradients_length': {0}".format(len(msg_payload['gradients']))
        elif msg.topic == params.MQTT_TOPIC_FAIL_DONE:
            pass
        else:
            pass

        logger.info(log_msg)

        if params.MODE_SYNCHRONIZATION:
            if msg_payload['episode'] not in chief.messages_received_from_workers:
                chief.messages_received_from_workers[msg_payload['episode']] = {}

            chief.messages_received_from_workers[msg_payload['episode']][msg_payload["worker_id"]] = (msg.topic, msg_payload)
            if len(chief.messages_received_from_workers[chief.episode_chief]) == params.NUM_WORKERS - chief.NUM_DONE_WORKERS:
                is_include_topic_success_done = False
                parameters_transferred = None
                worker_episode_reward_str = ""
                for worker_id in range(params.NUM_WORKERS):
                    if worker_id in chief.messages_received_from_workers[chief.episode_chief]:
                        topic, _msg_payload = chief.messages_received_from_workers[chief.episode_chief][worker_id]
                        chief.process_message(topic=topic, msg_payload=_msg_payload)

                        worker_episode_reward_str += "W{0}[{1:7.4f}/{2:7.4f}] ".format(
                            worker_id,
                            chief.messages_received_from_workers[chief.episode_chief][worker_id][1]['episode_reward'],
                            np.mean(chief.episode_reward_over_recent_episodes[worker_id])
                        )

                        if 'actor_objective' in msg_payload:
                            chief.save_results(
                                worker_id,
                                chief.messages_received_from_workers[chief.episode_chief][worker_id][1]['episode_reward'],
                                np.mean(chief.episode_reward_over_recent_episodes[worker_id]),
                                chief.messages_received_from_workers[chief.episode_chief][worker_id][1]['loss'],
                                np.mean(chief.loss_over_recent_episodes[worker_id]),
                                chief.messages_received_from_workers[chief.episode_chief][worker_id][1]['actor_objective'],
                                np.mean(chief.actor_objective_over_recent_episodes[worker_id])
                            )
                        else:
                            chief.save_results(
                                worker_id,
                                chief.messages_received_from_workers[chief.episode_chief][worker_id][1]['episode_reward'],
                                np.mean(chief.episode_reward_over_recent_episodes[worker_id]),
                                chief.messages_received_from_workers[chief.episode_chief][worker_id][1]['loss'],
                                np.mean(chief.loss_over_recent_episodes[worker_id])
                            )

                        if topic == params.MQTT_TOPIC_SUCCESS_DONE:
                            is_include_topic_success_done = True
                            if params.MODE_PARAMETERS_TRANSFER:
                                parameters_transferred = _msg_payload["parameters"]

                if is_include_topic_success_done:
                    transfer_msg = chief.get_transfer_ack_msg(parameters_transferred, msg_payload)
                    chief_mqtt_client.publish(topic=params.MQTT_TOPIC_TRANSFER_ACK, payload=transfer_msg, qos=0, retain=False)
                else:
                    grad_update_msg = chief.get_update_ack_msg(msg_payload=msg_payload)
                    chief_mqtt_client.publish(topic=params.MQTT_TOPIC_UPDATE_ACK, payload=grad_update_msg, qos=0, retain=False)

                chief.messages_received_from_workers[chief.episode_chief].clear()

                chief.save_graph()

                if params.NUM_WORKERS > 1 and params.NUM_WORKERS - chief.NUM_DONE_WORKERS > 1:
                    print("episode_chief:{0:3d} - {1}\n".format(chief.episode_chief, worker_episode_reward_str))
                chief.episode_chief += 1
        else:
            chief.process_message(msg.topic, msg_payload)

            if chief.num_messages == 0 or chief.num_messages % 200 == 0:
                chief.save_graph()

            chief.num_messages += 1
    except:
        traceback.print_exc()
        sys.exit(-1)


if __name__ == "__main__":
    chief_mqtt_client = mqtt.Client("dist_trans_chief")

    chief_mqtt_client.on_connect = on_chief_connect
    chief_mqtt_client.on_message = on_chief_message
    if params.MQTT_LOG:
        chief.on_log = on_chief_log

    chief_mqtt_client.connect(params.MQTT_SERVER, params.MQTT_PORT, keepalive=3600)

    chief_mqtt_client.loop_start()

    while True:
        try:
            time.sleep(1)
            if chief.NUM_DONE_WORKERS == params.NUM_WORKERS:
                chief_mqtt_client.loop_stop()
                break
        except KeyboardInterrupt as error:
            print("=== {0:>8} is aborted by keyboard interrupt".format('Chief'))
        except Exception as e:
            traceback.print_exc()
            sys.exit(-1)
