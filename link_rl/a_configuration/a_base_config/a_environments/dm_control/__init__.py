from link_rl.c_encoders.a_encoder import ENCODER


class ConfigDmControl:
    def __init__(self):
        self.ACTION_REPEAT = 4  # same to FRAME_SKIP
        self.IMG_SIZE = 84
        self.FRAME_STACK = 3
        self.FIXED_TOTAL_TIME_STEPS_PER_EPISODE = 1_000
        self.GRAY_SCALE = True
        self.FROM_PIXELS = False
        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value
