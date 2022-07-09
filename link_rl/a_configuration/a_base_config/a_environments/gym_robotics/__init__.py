from link_rl.c_encoders.a_encoder import ENCODER


class ConfigGymRobotics:
    def __init__(self):
        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value
