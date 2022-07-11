from link_rl.c_encoders.a_encoder import ENCODER


class ConfigGymAtari:
    def __init__(self):
        self.ENCODER_TYPE = ENCODER.NatureAtariEncoder.value
        self.FRAME_SKIP = 4


class ConfigGymBox2D:
    def __init__(self):
        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value


class ConfigGymClassicControl:
    def __init__(self):
        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value


class ConfigMujoco:
    def __init__(self):
        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value


class ConfigGymToyText:
    def __init__(self):
        self.ENCODER_TYPE = ENCODER.IdentityEncoder.value
