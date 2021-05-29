# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from codes.c_models.discrete_action.discrete_action_model import DiscreteActionModel
from codes.e_utils.names import DeepLearningModelName


class SimpleModel(DiscreteActionModel):
    def __init__(self, worker_id, input_shape=None, num_outputs=None, params=None, device=None):
        super(SimpleModel, self).__init__(worker_id, params, device)
        self.__name__ = "SimpleMlpModel"
        self.params = params

        num_inputs = input_shape[0]
        if params.DEEP_LEARNING_MODEL == DeepLearningModelName.SIMPLE_MLP:
            self.base = Simple_MLP_Base(
                num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
            )
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.SIMPLE_CNN:
            self.base = Simple_CNN_Base(
                input_shape=input_shape, num_outputs=num_outputs
            )
        elif params.DEEP_LEARNING_MODEL == DeepLearningModelName.SIMPLE_SMALL_CNN:
            self.base = Simple_SmallCNN_Base(
                input_shape=input_shape, num_outputs=num_outputs
            )
        else:
            raise ValueError()

        self.reset_average_gradients()

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)
        return self.base.forward(inputs)


class Simple_MLP_Base(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(Simple_MLP_Base, self).__init__()
        self.__name__ = "Simple_MLP_Base"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE

        self.net = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, num_outputs)
        )

        self.layers_info = {"net": self.net}

        self.train()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        if torch.is_tensor(x):
            x = x.to(torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32)

        return self.net(x)


class Simple_CNN_Base(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(Simple_CNN_Base, self).__init__()

        self.__name__ = "Simple_CNN_Base"

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )

        self.conv.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        self.layers_info = {"conv": self.conv, "fc": self.fc}

        self.train()

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        if torch.is_tensor(x):
            fx = x.to(torch.float32)
        else:
            fx = torch.tensor(x, dtype=torch.float32)

        conv_out = self.conv(fx).view(fx.size()[0], -1)
        out = self.fc(conv_out)
        return out


class Simple_SmallCNN_Base(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(Simple_SmallCNN_Base, self).__init__()

        self.__name__ = "Simple_SmallCNN_Base"

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        self.conv.apply(self.init_weights)
        self.fc_adv.apply(self.init_weights)

        self.layers_info = {"conv": self.conv, "fc": self.fc}

        self.train()

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight)

    def _get_conv_out(self, shape):
        o = self.conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        if torch.is_tensor(x):
            fx = x.to(torch.float32)
        else:
            fx = torch.tensor(x, dtype=torch.float32)

        conv_out = self.conv(fx).view(fx.size()[0], -1)
        out = self.fc(conv_out)
        return out