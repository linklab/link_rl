def manual_swing_up(self):
    print("\n***** Swing Up Start!!! *****")

    previousTime = time.perf_counter()
    last_pendulum_radian = 0
    motorPWM = 0

    while True:
        # if the difference between the current time and the last time an SPI transaction
        # occurred is greater than the sample time, start a new SPI transaction
        currentTime = time.perf_counter()
        if currentTime - previousTime >= UNIT_TIME:
            # print("|| Time difference: {0} s ||".format(currentTime - previousTime))

            previousTime = currentTime

            motor_radian, pendulum_radian = self.read_data()

            if 0.0 <= abs(pendulum_radian) <= PI * 11.0 / 180.0:
                break

            angular_variation = (pendulum_radian - last_pendulum_radian)
            # angular variation filtering
            if angular_variation > 2.5:
                angular_variation -= math.pi * 2
            elif angular_variation < -2.5:
                angular_variation += math.pi * 2

            pendulum_angular_velocity = angular_variation / UNIT_TIME

            last_pendulum_radian = pendulum_radian

            voltage = 80.0  # 48.65 # 49.215

            if abs(pendulum_angular_velocity) > 25:
                voltage /= int(10 * np.log(abs(pendulum_angular_velocity)))

            if PI >= abs(pendulum_radian) >= PI * 90.0 / 180.0:
                if pendulum_radian >= 0:
                    pendulum_radian = math.pi - pendulum_radian
                else:
                    pendulum_radian = - math.pi + abs(pendulum_radian)

                if pendulum_angular_velocity == 0:
                    if random.random() < 0.5:
                        motorPWM = int(-2 * math.cos(pendulum_radian) * voltage)
                    else:
                        motorPWM = int(2 * math.cos(pendulum_radian) * voltage)
                elif pendulum_angular_velocity < 0:
                    motorPWM = int(-2 * math.cos(pendulum_radian) * voltage)
                else:
                    motorPWM = int(2 * math.cos(pendulum_radian) * voltage)

            self.__set_motor_command(motorPWM, "blue")

    print("\n***** Swing Up complete!!! *****")


def manual_balance(self):
    theta_n_k1 = 0.0
    theta_dot_k1 = 0.0
    alpha_n_k1 = 0.0
    alpha_dot_k1 = 0.0

    kp_theta = 2.0
    kd_theta = -2.0
    kp_alpha = -30.0
    kd_alpha = 2.5

    previousTime = time.perf_counter()

    count = 0

    # time
    while count < 1500 / 5:
        # if the difference between the current time and the last time an SPI transaction
        # occurred is greater than the sample time, start a new SPI transaction
        currentTime = time.perf_counter()
        if currentTime - previousTime >= UNIT_TIME * 5:
            # print("|| Time difference: {0} s ||".format(currentTime - previousTime))

            previousTime = currentTime

            # LED Blue
            theta, alpha = self.read_data()

            # if the pendulum is within +/-30 degrees of upright, enable balance control
            if abs(alpha) <= (30.0 * math.pi / 180.0):
                # transfer function = 50s/(s+50)
                # z-transform at 1ms = (50z - 50)/(z-0.9512)
                theta_n = -theta
                theta_dot = (50.0 * theta_n) - (50.0 * theta_n_k1) + (0.7612 * theta_dot_k1)
                theta_n_k1 = theta_n
                theta_dot_k1 = theta_dot

                # transfer function = 50s/(s+50)
                # z-transform at 1ms = (50z - 50)/(z-0.9512)
                alpha_n = -alpha
                alpha_dot = (50.0 * alpha_n) - (50.0 * alpha_n_k1) + (0.7612 * alpha_dot_k1)
                alpha_n_k1 = alpha_n
                alpha_dot_k1 = alpha_dot

                # multiply by proportional and derivative gains
                motorVoltage = (theta * kp_theta) + (theta_dot * kd_theta) + (alpha * kp_alpha) + (
                        alpha_dot * kd_alpha)

                # set the saturation limit to +/- 15V
                if motorVoltage > 15.0:
                    motorVoltage = 15.0
                elif motorVoltage < -15.0:
                    motorVoltage = -15.0

                # invert for positive CCW
                motorVoltage = -motorVoltage

                # convert the analog value to the PWM duty cycle that will produce the same average voltage
                motorPWM = int(motorVoltage * (625.0 / 15.0))
                if motorPWM > 280:
                    motorPWM = 280
                elif motorPWM < -280:
                    motorPWM = -280

                # print(motorPWM)

                self.__set_motor_command(motorPWM, "green")

                count += 1

            else:
                self.read_data()
                break
    self.last_motor_radian = theta
    self.last_pendulum_radian = alpha
    self.is_reset = False
