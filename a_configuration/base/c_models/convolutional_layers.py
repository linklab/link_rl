from g_utils.types import ModelType


class ParameterConvolutionalLayer:
    MODEL_TYPE = ModelType.CONVOLUTIONAL


class ParameterSmallConvolutionalLayer(ParameterConvolutionalLayer):
    OUT_CHANNELS_PER_LAYER = [8, 16, 32]


class ParameterMediumConvolutionalLayer(ParameterConvolutionalLayer):
    OUT_CHANNELS_PER_LAYER = [16, 32, 64, 64]


class ParameterLargeConvolutionalLayer(ParameterConvolutionalLayer):
    OUT_CHANNELS_PER_LAYER = [32, 64, 64, 128]
