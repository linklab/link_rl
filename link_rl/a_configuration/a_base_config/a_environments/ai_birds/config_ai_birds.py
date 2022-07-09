from link_rl.c_encoders.a_encoder import ENCODER


class ConfigAiBirds:
    def __init__(self):
        self.ENV_NAME = "AiBirds_v0"
        self.EPISODE_REWARD_MIN_SOLVED = 10000.0
        self.FROM_PIXEL = True
        self.ENCODER_TYPE = ENCODER.SimpleConvEncoder.value

