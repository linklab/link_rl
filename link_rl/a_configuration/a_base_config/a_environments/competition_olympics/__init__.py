from link_rl.c_encoders.a_encoder import ENCODER


class ConfigCompetitionOlympics:
    def __init__(self):
        self.IMG_SIZE = 40
        self.FROM_PIXELS = True
        self.ENCODER_TYPE = ENCODER.SimpleConvEncoder.value
