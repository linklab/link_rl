from g_utils.types import ModelType


class ConfigRecurrentConvolutionalModel:
    def __init__(self, model_type):
        if model_type == ModelType.SMALL_RECURRENT_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [8, 16, 32]
            self.KERNEL_SIZE_PER_LAYER = [4, 3, 2]
            self.STRIDE_PER_LAYER = [2, 2, 1]

            self.NEURONS_PER_REPRESENTATION_LAYER = [128]

            self.HIDDEN_SIZE = 128
            self.NUM_LAYERS = 1

            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128]  # X
            # self.NEURONS_PER_FULLY_CONNECTED_LAYER = []  # O

        elif model_type == ModelType.MEDIUM_RECURRENT_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [16, 32, 64, 64]
            self.KERNEL_SIZE_PER_LAYER = [8, 4, 4, 3]
            self.STRIDE_PER_LAYER = [4, 2, 2, 1]

            self.NEURONS_PER_REPRESENTATION_LAYER = [128]

            self.HIDDEN_SIZE = 128
            self.NUM_LAYERS = 3

            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128]

        elif model_type == ModelType.LARGE_RECURRENT_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [32, 64, 64, 128]
            self.KERNEL_SIZE_PER_LAYER = [4, 4, 4, 3]
            self.STRIDE_PER_LAYER = [2, 2, 2, 1]

            self.NEURONS_PER_REPRESENTATION_LAYER = [256]

            self.HIDDEN_SIZE = 256
            self.NUM_LAYERS = 5

            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128]
        else:
            raise ValueError()
