# https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%208/Ch8_book.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class CuriosityMlpStateEncoder(nn.Module): #A
    def __init__(self, num_inputs, encoded_state_size=8):
        super(CuriosityMlpStateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.GELU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.GELU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.GELU(),
            nn.Linear(self.hidden_3_size, encoded_state_size)
        )
        self.encoded_state_size = encoded_state_size

    def forward(self, x):
        y = self.encoder(x)
        return y


class CuriosityCnnStateEncoder(nn.Module): #A
    def __init__(self, input_shape):
        super(CuriosityCnnStateEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.GELU(),
        )

        self.encoded_state_size = self._get_conv_out(input_shape)

    def _get_conv_out(self, shape):
        o = self.actor_conv(Variable(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.normalize(x)
        y = self.encoder(x)
        y = y.flatten(start_dim=1) #size N, 288
        return y


class CuriosityForwardModel(nn.Module): #C
    def __init__(self, encoder_state_size, num_outputs):
        super(CuriosityForwardModel, self).__init__()
        self.linear1 = nn.Linear(encoder_state_size + num_outputs, 256)
        self.linear2 = nn.Linear(256, encoder_state_size)
        self.num_outputs = num_outputs

    def forward(self, state, action):  # 다음 상태 예측
        action_ = torch.zeros(action.shape[0], self.num_outputs) #D
        indices = torch.stack((torch.arange(action.shape[0]), action.squeeze()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat((state, action_), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y


class CuriosityInverseModel(nn.Module): #B
    def __init__(self, encoded_state_size, num_outputs):
        super(CuriosityInverseModel, self).__init__()
        self.linear1 = nn.Linear(encoded_state_size * 2, 256)
        self.linear2 = nn.Linear(256, num_outputs)

    def forward(self, encoded_state, encoded_next_state): # 상태와 다음 상태를 산출할 수 있는 행동 예측
        x = torch.cat((encoded_state, encoded_next_state), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y, dim=1)
        return y


