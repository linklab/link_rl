from link_rl.g_utils.types import ModelType


class ConfigDmControl:
    def __init__(self):
        self.FROM_PIXELS = False
        self.MODEL_TYPE_PIXEL = ModelType.SMALL_2D_CONVOLUTIONAL