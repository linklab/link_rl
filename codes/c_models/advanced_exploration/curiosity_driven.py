# https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/blob/master/Chapter%208/Ch8_book.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class CuriosityMlpStateEncoder(nn.Module): #A
    def __init__(self, num_inputs, encoded_state_size=32, params=None, device=None):
        super(CuriosityMlpStateEncoder, self).__init__()

        self.params = params
        self.device = device

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.GELU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.GELU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.GELU(),
            nn.Linear(self.hidden_3_size, encoded_state_size)
        ).to(self.device)

        self.encoded_state_size = encoded_state_size

    def forward(self, x):
        y = self.encoder(x)
        return y


class CuriosityCnnStateEncoder(nn.Module): #A
    def __init__(self, observation_shape, params, device):
        super(CuriosityCnnStateEncoder, self).__init__()

        self.params = params
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GELU(),
        ).to(self.device)

        self.encoded_state_size = self._get_conv_out(observation_shape)

    def _get_conv_out(self, shape):
        o = self.encoder(Variable(torch.zeros(1, *shape)).to(self.device))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)
        x = F.normalize(x)
        y = self.encoder(x)
        y = y.flatten(start_dim=1) #size N, 288
        return y


class CuriosityForwardModel(nn.Module): #C
    def __init__(self, encoder_state_size, action_n, device):
        super(CuriosityForwardModel, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(encoder_state_size + action_n, 256).to(self.device)
        self.linear2 = nn.Linear(256, encoder_state_size).to(self.device)
        self.action_n = action_n

    def forward(self, state, action):  # ?????? ?????? ??????
        action = action.to(self.device)
        action_ = torch.zeros(action.shape[0], self.action_n).to(self.device) #D
        indices = torch.stack((torch.arange(action.shape[0]).to(self.device), action.squeeze()), dim=0)
        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat((state, action_), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y


class CuriosityInverseModel(nn.Module): #B
    def __init__(self, encoded_state_size, action_n, device):
        super(CuriosityInverseModel, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(encoded_state_size * 2, 256).to(self.device)
        self.linear2 = nn.Linear(256, action_n).to(self.device)

    def forward(self, encoded_state, encoded_next_state): # ????????? ?????? ????????? ????????? ??? ?????? ?????? ??????
        x = torch.cat((encoded_state, encoded_next_state), dim=1).to(self.device)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y, dim=1)
        return y


def intrinsic_curiosity_module_errors(
        curiosity_state_encoder, curiosity_forward_model, curiosity_inverse_model,
        state, action, next_state, forward_scale=1.0, inverse_scale=1.0e4
):
    # encoded_state.shape: [32, 8]
    # encoded_next_state.shape: [32, 8]
    encoded_state = curiosity_state_encoder(state)
    encoded_next_state = curiosity_state_encoder(next_state)

    # encoded_next_state_pred.shape: [32, 8]
    encoded_next_state_pred = curiosity_forward_model(encoded_state.detach(), action.detach())
    # forward_loss.shape: [32, 1]
    forward_loss = F.mse_loss(
        encoded_next_state_pred, encoded_next_state.detach(), reduction="none"
    ).sum(dim=-1).unsqueeze(dim=-1)
    forward_pred_loss = forward_scale * forward_loss

    # action_pred.shape: [32, 2]
    action_pred = curiosity_inverse_model(encoded_state, encoded_next_state)
    # inverse_loss.shape: [32, 1]
    inverse_loss = F.cross_entropy(action_pred, action.detach().flatten(), reduction="none").unsqueeze(dim=-1)
    inverse_pred_loss = inverse_scale * inverse_loss

    # forward_pred_loss.shape: [32, 1]
    # inverse_pred_loss.shape: [32, 1]
    return forward_pred_loss, inverse_pred_loss

