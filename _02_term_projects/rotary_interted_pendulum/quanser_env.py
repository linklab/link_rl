from quanser.hardware import MAX_STRING_LENGTH, HILError
from array import array
import copy
import math
import time

import gymnasium as gym
import numpy as np
import torch
from collections import deque

np.set_printoptions(precision=5, suppress=True)

ENCODER_CPR = 2048.0
RAD_PER_COUNT = 2.0 * math.pi / ENCODER_CPR


class QuanserEnv(gym.Env):
    def __init__(self, card):
        self.card = card

        # PWM mode setup
        self.card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)
        input_channels = array('I', [1])
        output_channels = array('I', [0])
        self.card.set_digital_directions(
            input_channels, len(input_channels),
            output_channels, len(output_channels),
        )
        self.card.write_digital(array('I', [0]), 1, array('I', [1]))

        # channels
        self.pwm_ch = array('I', [0])
        self.motor_enc_ch = array('I', [0])
        self.pend_enc_ch = array('I', [1])
        self.tach_ch = array('I', [14001])
        self.tach_motor_ch = array('I', [14000])

        # LED channels
        self.led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

        # encoder value buffers
        self.motor_enc_val = array('i', [0])
        self.pend_enc_val = array('i', [0])
        self.tach_vel_val = array('d', [0.0])
        self.tach_motor_vel_val = array('d', [0.0])

        # control params
        self.Ts = 0.006
        self.max_steps = 2000
        self.action_scale = 0.35

        # counters
        self.time_steps = 0
        self.step_count = 0
        self.reset_count = 0

        # PD gains
        self.Kp = 0.8
        self.Kd = 0.02

        self.last_action = 0.0
        self.std_error = deque(maxlen=2000)
        self.pen_init_count_list = []

        low_obs = np.array([
            -2.5, -1.0, -1.0, -np.inf, -np.inf,
        ], dtype=np.float32)
        high_obs = np.array([
            2.5, 1.0, 1.0, np.inf, np.inf,
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.init_count = 0
        self.pend_init_count = 0
        self._reset_init_count()

        self.step_time = None

    def _get_motor_angle(self):
        self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
        count = self.motor_enc_val[0] - self.init_count
        return count * RAD_PER_COUNT

    def _get_pendulum_angle(self):
        """[-π, π] 범위의 진자 각도를 반환한다. 6시 방향이 0."""
        self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
        raw_angle = (self.pend_enc_val[0] - self.pend_init_count) * RAD_PER_COUNT
        return ((raw_angle + math.pi) % (2 * math.pi)) - math.pi

    def _get_motor_velocity(self):
        self.card.read_other(self.tach_motor_ch, 1, self.tach_motor_vel_val)
        return self.tach_motor_vel_val[0] * RAD_PER_COUNT

    def _get_pendulum_velocity(self):
        self.card.read_other(self.tach_ch, 1, self.tach_vel_val)
        return self.tach_vel_val[0] * RAD_PER_COUNT

    def _reset_init_count(self):
        print("calibrate motor init count...")

        for duty_val in (-0.06, 0.06):
            for _ in range(301):
                self.card.write_pwm(self.pwm_ch, 1, array('d', [duty_val]))
                time.sleep(0.01)
            self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
            if duty_val < 0:
                push_max_count = self.motor_enc_val[0]
            else:
                push_min_count = self.motor_enc_val[0]

        self.init_count = (push_max_count + push_min_count) // 2
        print("set init count:", self.init_count)

    def _reset_pendulum_init_count(self) -> None:
        self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
        self.pen_init_count_list.append(copy.deepcopy(self.pend_enc_val[0]))
        if len(self.pen_init_count_list) > 100:
            pend_init_count_mean = int(np.mean(self.pen_init_count_list))
            print("pendulum init count diff:",
                  (self.pend_init_count - pend_init_count_mean) % int(ENCODER_CPR))
            self.pend_init_count = pend_init_count_mean

    def _get_pend_spin_num(self):
        return (self.pend_init_count - self.pend_enc_val[0]) / ENCODER_CPR

    def get_init_observations(self):
        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        motor_vel = self._get_motor_velocity()
        pend_vel = self._get_pendulum_velocity()

        return torch.tensor([
            motor_angle, math.sin(pend_angle), math.cos(pend_angle),
            motor_vel, pend_vel,
        ], dtype=torch.float32)

    def normalize_observation(self, observation):
        return torch.tensor([
            observation[0] / 1.8,
            observation[1],
            observation[2],
            observation[3] / 20.0,
            observation[4] / 40.0,
        ], dtype=torch.float32)

    def _set_led(self, r: float, g: float, b: float):
        values = np.array([r, g, b], dtype=np.float64)
        self.card.write_other(self.led_channels, len(self.led_channels), values)

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        self.step_time = None
        self.last_action = 0.0
        self.std_error = deque(maxlen=2000)

        print("\n======RESET START======")
        self._set_led(0.0, 0.0, 1.0)

        start = time.time()
        reset_counter = 0
        reset_success_num = 0

        if self.reset_count % 5 == 0:
            self._reset_init_count()

        thresh_pend_vel = 0.1
        thresh_error_rad = 0.1

        while True:
            reset_counter += 1
            cur_rad = self._get_motor_angle()
            error_rad = -cur_rad
            pend_vel = self._get_pendulum_velocity()

            if abs(error_rad) < thresh_error_rad and abs(pend_vel) < thresh_pend_vel:
                reset_success_num += 1
            else:
                reset_success_num = 0
                self.pen_init_count_list = []

            if reset_success_num > 50:
                self.card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
                break

            omega = self._get_motor_velocity()
            duty = np.clip(self.Kp * 2 * error_rad - self.Kd * omega, -0.04, 0.04)
            self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
            time.sleep(0.005)

            if reset_counter > 1000 and reset_success_num == 0:
                thresh_pend_vel += 0.05
                thresh_error_rad += 0.05
                reset_counter = 0
                if abs(error_rad) > 0.3:
                    self._reset_init_count()
                    print("RESET FAILED, RE-CALIBRATE MOTOR INIT COUNT")
                else:
                    print(f"FAILED TO RESET, RETRYING... "
                          f"error_rad: {abs(error_rad):.3f}, "
                          f"pend_vel: {abs(pend_vel):.3f}")

        self.pen_init_count_list = []
        for _ in range(101):
            self._reset_pendulum_init_count()
            time.sleep(0.003)

        print("\n======RESET END======")
        print(f"Reset time: {time.time() - start:.2f} sec")

        obs = self.normalize_observation(self.get_init_observations())
        self.step_time = time.perf_counter()
        self._set_led(1.0, 0.0, 0.0)
        return obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        self.time_steps += 1
        self.step_count += 1
        self.last_action = actions.item()

        # read observations
        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        motor_vel = self._get_motor_velocity()
        pend_vel = self._get_pendulum_velocity()

        next_obs = torch.tensor([
            motor_angle, math.sin(pend_angle), math.cos(pend_angle),
            motor_vel, pend_vel,
        ], dtype=torch.float32)
        next_obs = self.normalize_observation(next_obs)

        # reward
        reward = (pend_angle / math.pi) ** 2
        if abs(pend_angle) > 2.96706:  # 170 deg
            reward *= 2
        reward += 0.1

        # termination
        terminated = False
        pend_spin_num = self._get_pend_spin_num()
        if abs(pend_spin_num) > 5.0:
            print("PENDULUM SPIN OVER:", pend_spin_num)
            terminated = True
        if abs(math.degrees(motor_angle)) > 90.0:
            print("MOTOR ANGLE OVER:", math.degrees(motor_angle))
            terminated = True
        if abs(pend_vel) > 40.0:
            print("PENDULUM VELOCITY OVER:", pend_vel)
            terminated = True

        truncated = self.step_count >= self.max_steps
        if truncated:
            print("TRUNCATED")

        if terminated or truncated:
            self._set_led(0.0, 0.0, 1.0)
            self.reset_helper()

        return next_obs, reward, terminated, truncated, {}
    
    def reset_helper(self):
        reset_counter = 0
        reset_success_num = 0

        while reset_counter < 3000:
            reset_counter += 1
            self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
            cur_rad = (self.motor_enc_val[0] - self.init_count) * RAD_PER_COUNT
            error_rad = -cur_rad

            if abs(error_rad) < 0.2:
                reset_success_num += 1
            else:
                reset_success_num = 0

            if reset_success_num > 150:
                self.card.write_pwm(self.pwm_ch, 1, array('d', [0.0]))
                break

            omega = self._get_motor_velocity()

            if cur_rad > 1.57:
                duty = -0.4
            elif cur_rad < -1.57:
                duty = 0.4
            else:
                duty = np.clip(self.Kp * error_rad - self.Kd * omega, -0.2, 0.2)

            self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
            time.sleep(0.005)

    def apply_action(self, actions):
        pwm = float(actions.item()) * self.action_scale
        self.card.write_pwm(self.pwm_ch, 1, array('d', [pwm]))
