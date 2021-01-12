import paho.mqtt.client as paho
import threading
import random
# import spidev
import time
import math
import numpy as np
from datetime import datetime
import json
import datetime
import spidev
import struct

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 5000

MQTT_SERVER = '192.168.0.10'

MQTT_SUB_FROM_ENV_POWER = 'motor_power'
MQTT_SUB_RESET = 'reset'
MQTT_PUB_RESET_COMPLETE = 'reset_complete'
MQTT_PUB_TO_ENV = 'next_state'
MQTT_ERROR = 'error'

headers = {"content-type": "application/json"}

last_time = time.time()

class DoubleRotaryInvertedPendulum:
	def __init__(self):
		self.mqtt_client = paho.Client(client_id="DoubleRIP")
		self.mqtt_client.on_connect = self.on_connect
		self.mqtt_client.on_message = self.on_message
		self.mqtt_client.connect(MQTT_SERVER, 1883, 3600)
                self.mqtt_client.loop_start()
                self.done = True

	@staticmethod
	def on_connect(client, userdata, flags, rc):
                print("mqtt broker connected with result code " + str(rc) + "\n")
		client.subscribe(topic=MQTT_SUB_FROM_ENV_POWER)
		client.subscribe(topic=MQTT_SUB_RESET)
                client.subscribe(topic=MQTT_ERROR)

	def on_message(self, client, useradta, msg):
                if msg.topic == MQTT_SUB_RESET:
                        print("Sub Reset Topic")
                        self.reset()
		elif msg.topic == MQTT_SUB_FROM_ENV_POWER:
                        # print("Sub Step Topic")
			motor_power = int(msg.payload.decode("utf-8"))
                        print("motor_power :", motor_power)
                        self.step(motor_power)
                elif msg.topic == MQTT_ERROR:
                        print("STOP")
                        self.stop()
        
        def reset(self):
            arm_angle, arm_vel, link_angle, link_vel = self.calculate_state()
            
            print("arm angle :", arm_angle)
            print("arm vel :", arm_vel)
            print("link angle :", link_angle)
            print("link vel :", link_vel)
            
	    self.mqtt_client.publish(
	    	MQTT_PUB_RESET_COMPLETE,
            "{0}|{1}|{2}|{3}".format(arm_angle, arm_vel, link_angle, link_vel)
            )
        
        def step(self, motor_power):
                if motor_power > 0:
                    action_1, action_2 = self.calculate_action(motor_power)
                    spi.xfer2([0x40, 0x00, 0x02, action_1, action_2])
                else:
                    motor_power = -motor_power
                    action_1, action_2 = self.calculate_action(motor_power)
                    spi.xfer2([0x40, 0x00, 0x03, action_1, action_2])
                
                self.pub_next_state()
                
                arm_angle, arm_vel, link_angle, link_vel = self.calculate_state()
            
                print("arm angle :", arm_angle)
                print("arm vel :", arm_vel)
                print("link angle :", link_angle)
                print("link vel :", link_vel)
                
                     
        def pub_next_state(self):
                arm_angle, arm_vel, link_angle, link_vel = self.calculate_state()
                #print("!!!!!!!!!!!",arm_angle, arm_vel, link_angle, link_vel)
                self.mqtt_client.publish(
                    MQTT_PUB_TO_ENV,
                    "{0}|{1}|{2}|{3}".format(arm_angle, arm_vel, link_angle, link_vel)
                    )
                # print("@@@ Next State Publish @@@")
                
        def calculate_action(self, motor_power):
                motor_power_hex = hex(motor_power)
                motor_power_str = str(motor_power_hex)
                if len(motor_power_str) < 5:
                    front_hex = 0x00
                    back_hex = int(motor_power_str, 16)
                else:
                    front_hex_str = '0x0' + motor_power_str[2]
                    back_hex_str = '0x' + motor_power_str[3:]
                    front_hex = int(front_hex_str, 16)
                    back_hex = int(back_hex_str, 16)
                    
                return front_hex, back_hex
            
        def calculate_state(self):
                data = spi.xfer2([128 if i==0 else i for i in range(21)])

                t = float((data[1]<<24)+(data[2]<<16)+(data[3]<<8)+data[4])
                arm_angle = float((data[5]<<24)+(data[6]<<16)+(data[7]<<8)+data[8])
                arm_vel = float((data[9]<<24)+(data[10]<<16)+(data[11]<<8)+data[12])
                link_angle = float((data[13]<<24)+(data[14]<<16)+(data[15]<<8)+data[16])
                link_vel = float((data[17]<<24)+(data[18]<<16)+(data[19]<<8)+data[20])                
                
                t = t/1000
                arm_angle = -(4294967296-arm_angle)/100 if arm_angle > 4000000000 else arm_angle/100
                arm_vel = -(4294967296-arm_vel)/100 if arm_vel > 4000000000 else arm_vel/100
                link_angle = -(4294967296-link_angle)/100 if link_angle > 4000000000 else link_angle/100
                link_vel = -(4294967296-link_vel)/100 if link_vel > 4000000000 else link_vel/100
                
                return arm_angle, arm_vel, link_angle, link_vel
            
        def start(self):
            spi.xfer2([
                    0x40,0x00,0x00
                    ])
            spi.xfer2([
                    0x40,0x00,0x10, 0x00,0x00
                    ])
            spi.xfer2([
                    0x40,0x00,0x00
                    ])
            spi.xfer2([
                    0x40,0x00,0x10,0x00,0x00
                    ])
            spi.xfer2([
                    0x40,0x00,0x00
                    ])
            print("START!!!!")
            arm_angle, arm_vel, link_angle, link_vel = self.calculate_state()
            
            print("arm angle :", arm_angle)
            print("arm vel :", arm_vel)
            print("link angle :", link_angle)
            print("link vel :", link_vel)
        
        def stop(self):
            spi.xfer2([0x40,0x00,0x10, 0x00,0x00])
            spi.xfer2([0x40,0x00,0x01])
                
if __name__ == "__main__":
	print("Main Start!!!!")
	try:
                drip = DoubleRotaryInvertedPendulum()
                drip.start()
                while True:
                    time.sleep(0.1)
                
	except Exception as error:
                print(error)
                drip.mqtt_client.publish(topic=MQTT_ERROR, payload="IOError")
                drip.stop()
                
