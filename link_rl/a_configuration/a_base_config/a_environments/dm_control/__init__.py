from link_rl.g_utils.types import ModelType


class ConfigDmControl:
    def __init__(self):
        self.FROM_PIXELS = False
        self.MODEL_TYPE_PIXEL = ModelType.SMALL_2D_CONVOLUTIONAL
        self.ACTION_REPEAT = 4
        self.IMG_SIZE = 84
        self.FRAME_STACK = 3
        self.FIXED_TOTAL_TIME_STEPS_PER_EPISODE = 1_000

