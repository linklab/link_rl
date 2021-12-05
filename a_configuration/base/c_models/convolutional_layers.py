from g_utils.types import ModelType


class ParameterConvolutionalLayer:
    def __init__(self):
        self.MODEL_TYPE = ModelType.CONVOLUTIONAL


class ParameterSmallConvolutionalLayer(ParameterConvolutionalLayer):
    def __init__(self):
        super(ParameterSmallConvolutionalLayer, self).__init__()
        self.OUT_CHANNELS_PER_LAYER = [8, 16, 32]
        self.KERNEL_SIZE_PER_LAYER = [4, 3, 2]
        self.STRIDE_PER_LAYER = [2, 2, 1]
        self.NEURONS_PER_FULLY_CONNECTED_LAYER = [128]


class ParameterMediumConvolutionalLayer(ParameterConvolutionalLayer):
    def __init__(self):
        super(ParameterMediumConvolutionalLayer, self).__init__()
        self.OUT_CHANNELS_PER_LAYER = [16, 32, 64, 64]
        self.KERNEL_SIZE_PER_LAYER = [8, 4, 4, 3]
        self.STRIDE_PER_LAYER = [4, 2, 2, 1]
        self.NEURONS_PER_FULLY_CONNECTED_LAYER = [256]


class ParameterLargeConvolutionalLayer(ParameterConvolutionalLayer):
    def __init__(self):
        super(ParameterLargeConvolutionalLayer, self).__init__()
        self.OUT_CHANNELS_PER_LAYER = [32, 64, 64, 128]
        self.KERNEL_SIZE_PER_LAYER = [4, 4, 4, 3]
        self.STRIDE_PER_LAYER = [2, 2, 2, 1]
        self.NEURONS_PER_FULLY_CONNECTED_LAYER = [512, 256]
