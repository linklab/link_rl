from link_rl.c_encoders.a_encoder import ENCODER


class ConfigCompetitionOlympics:
    def __init__(self):
        self.FROM_PIXEL = True
        self.ENCODER_TYPE = ENCODER.SimpleConvEncoder.value
