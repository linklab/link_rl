# -*- coding:utf-8 -*-
import pickle
import time
import zlib
import paho.mqtt.client as mqtt
import sys, os

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from config.parameters import PARAMETERS as params

from common.logger import get_logger
from common.chief_workers.worker import Worker

worker_id = int(sys.argv[1])
logger = get_logger("worker_{0}".format(worker_id))


def on_worker_log(mqttc, obj, level, string):
    print(string)


def on_worker_connect(client, userdata, flags, rc):
    if rc == 0:
        msg = "Worker {0} is successfully connected with broker@{1}".format(worker_id, params.MQTT_SERVER)
        logger.info(msg)
        client.subscribe(params.MQTT_TOPIC_TRANSFER_ACK)
        client.subscribe(params.MQTT_TOPIC_UPDATE_ACK)
        print(msg)


def on_worker_message(client, userdata, msg):
    msg_payload = zlib.decompress(msg.payload)
    msg_payload = pickle.loads(msg_payload)

    if msg.topic == params.MQTT_TOPIC_UPDATE_ACK:
        log_msg = "[RECV] TOPIC: {0}, PAYLOAD: 'episode_chief': {1}".format(
            msg.topic,
            msg_payload['episode_chief']
        )

        if params.MODE_GRADIENTS_UPDATE:
            log_msg += ", avg_grad_length: {0} \n".format(
                len(msg_payload['avg_gradients'])
            )
        else:
            log_msg += "\n"

        logger.info(log_msg)

        if not worker.is_success_or_fail_done and params.MODE_GRADIENTS_UPDATE:
            worker.update_process(msg_payload['avg_gradients'])

        worker.episode_chief = msg_payload["episode_chief"]
        print("Update_Ack: " + worker.episode_chief)
        
    elif msg.topic == params.MQTT_TOPIC_TRANSFER_ACK:
        log_msg = "[RECV] TOPIC: {0}, PAYLOAD: 'episode_chief': {1}".format(
            msg.topic,
            msg_payload['episode_chief']
        )

        if params.MODE_PARAMETERS_TRANSFER:
            log_msg += ", parameters_length: {0} \n".format(
                len(msg_payload['parameters'])
            )
        else:
            log_msg += "\n"

        logger.info(log_msg)

        if not worker.is_success_or_fail_done and params.MODE_PARAMETERS_TRANSFER:
            worker.transfer_process(msg_payload['parameters'])

        worker.episode_chief = msg_payload["episode_chief"]
        print("Transfer_Ack: " + worker.episode_chief)

    else:
        print("pass")
        pass


if __name__ == "__main__":
    worker_mqtt_client = mqtt.Client("rl_worker_{0}".format(worker_id))
    worker_mqtt_client.on_connect = on_worker_connect
    worker_mqtt_client.on_message = on_worker_message
    if params.MQTT_LOG:
        worker_mqtt_client.on_log = on_worker_log

    worker_mqtt_client.connect(params.MQTT_SERVER, params.MQTT_PORT, keepalive=3600)
    worker_mqtt_client.loop_start()

    stderr = sys.stderr
    sys.stderr = sys.stdout
    try:
        worker = Worker(logger, worker_id, worker_mqtt_client, params)
        worker.start_train()

        time.sleep(1)
        worker_mqtt_client.loop_stop()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Worker {0}'.format(worker_id)))
    finally:
        sys.stderr = stderr
