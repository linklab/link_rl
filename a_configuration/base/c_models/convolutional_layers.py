from g_utils.types import ModelType


class ParameterConvolutionalLayer:
    MODEL_TYPE = ModelType.CONVOLUTIONAL


class ParameterSmallConvolutionalLayer(ParameterConvolutionalLayer):
    OUT_CHANNELS_PER_LAYER = [8, 16, 32]
    KERNEL_SIZE_PER_LAYER = [4, 3, 2]
    STRIDE_PER_LAYER = [2, 2, 1]
    NEURONS_PER_FULLY_CONNECTED_LAYER = [128]


class ParameterMediumConvolutionalLayer(ParameterConvolutionalLayer):
    OUT_CHANNELS_PER_LAYER = [16, 32, 64, 64]
    KERNEL_SIZE_PER_LAYER = [8, 4, 4, 3]
    STRIDE_PER_LAYER = [4, 2, 2, 1]
    NEURONS_PER_FULLY_CONNECTED_LAYER = [256]


class ParameterLargeConvolutionalLayer(ParameterConvolutionalLayer):
    OUT_CHANNELS_PER_LAYER = [32, 64, 64, 128]
    KERNEL_SIZE_PER_LAYER = [4, 4, 4, 3]
    STRIDE_PER_LAYER = [2, 2, 2, 1]
    NEURONS_PER_FULLY_CONNECTED_LAYER = [512, 256]
