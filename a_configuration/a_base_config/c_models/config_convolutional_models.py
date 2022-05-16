from g_utils.types import ModelType


class Config1DConvolutionalModel:
    def __init__(self, model_type):
        if model_type == ModelType.TINY_1D_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [48]
            self.KERNEL_SIZE_PER_LAYER = [2]
            self.STRIDE_PER_LAYER = [1]
            self.PADDING = 0

            self.NEURONS_PER_REPRESENTATION_LAYER = [64]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128]

        elif model_type == ModelType.SMALL_1D_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [8, 16, 32]
            self.KERNEL_SIZE_PER_LAYER = [4, 3, 2]
            self.STRIDE_PER_LAYER = [1, 1, 1]
            self.PADDING = 0

            self.NEURONS_PER_REPRESENTATION_LAYER = [128]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128]  # X
            # self.NEURONS_PER_FULLY_CONNECTED_LAYER = []  # O

        elif model_type == ModelType.MEDIUM_1D_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [16, 32, 64, 64]
            self.KERNEL_SIZE_PER_LAYER = [8, 4, 4, 3]
            self.STRIDE_PER_LAYER = [1, 1, 1, 1]
            self.PADDING = 0

            self.NEURONS_PER_REPRESENTATION_LAYER = [256]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256]

        elif model_type == ModelType.LARGE_1D_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [32, 64, 64, 128]
            self.KERNEL_SIZE_PER_LAYER = [4, 4, 4, 3]
            self.STRIDE_PER_LAYER = [1, 1, 1, 1]
            self.PADDING = 0

            self.NEURONS_PER_REPRESENTATION_LAYER = [512]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256]

        else:
            raise ValueError()


class Config2DConvolutionalModel:
    def __init__(self, model_type):
        if model_type == ModelType.TINY_2D_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [4, 8]
            self.KERNEL_SIZE_PER_LAYER = [2, 2]
            self.STRIDE_PER_LAYER = [1, 1]
            self.PADDING = 0

            self.NEURONS_PER_REPRESENTATION_LAYER = [64]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [64]

        elif model_type == ModelType.SMALL_2D_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [8, 16, 32]
            self.KERNEL_SIZE_PER_LAYER = [4, 3, 2]
            self.STRIDE_PER_LAYER = [2, 2, 1]
            self.PADDING = 0

            self.NEURONS_PER_REPRESENTATION_LAYER = [128]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128]  # X
            # self.NEURONS_PER_FULLY_CONNECTED_LAYER = []  # O

        elif model_type == ModelType.MEDIUM_2D_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [16, 32, 64, 64]
            self.KERNEL_SIZE_PER_LAYER = [4, 4, 4, 3]
            self.STRIDE_PER_LAYER = [2, 2, 2, 1]
            self.PADDING = 0

            self.NEURONS_PER_REPRESENTATION_LAYER = [256]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256]

        elif model_type == ModelType.LARGE_2D_CONVOLUTIONAL:
            self.OUT_CHANNELS_PER_LAYER = [32, 64, 64, 128]
            self.KERNEL_SIZE_PER_LAYER = [4, 4, 4, 3]
            self.STRIDE_PER_LAYER = [2, 2, 2, 1]
            self.PADDING = 0

            self.NEURONS_PER_REPRESENTATION_LAYER = [512]
            self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256]

        else:
            raise ValueError()
