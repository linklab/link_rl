import os
import time

for _ in range(10):
    os.system('mosquitto_pub -h 127.0.0.1 -p 1883 -t "motor_power" -m "-150|action|1"')
    time.sleep(0.05)
    os.system('mosquitto_pub -h 127.0.0.1 -p 1883 -t "motor_power" -m "150|action|1"')
    time.sleep(0.05)

os.system('mosquitto_pub -h 127.0.0.1 -p 1883 -t "motor_power" -m "0|action|1"')