# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn

from codes.c_models.base_model import BaseModel


class DuelingDQNModel(BaseModel):
    def __init__(self, s_size, a_size, worker_id, params, device):
        super(DuelingDQNModel, self).__init__(s_size, a_size, worker_id, params, device)
        self.__name__ = "DuelingDQNModel"

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.base = DuelingMLPBase(
            num_inputs=s_size,
            num_ouputs=a_size,
            params=self.params
        )

        self.reset_average_gradients()

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)
        return self.base.forward(inputs)


class DuelingMLPBase(nn.Module):
    def __init__(self, num_inputs, num_ouputs, params):
        super(DuelingMLPBase, self).__init__()
        self.__name__ = "DuelingMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.net = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU()
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, num_ouputs)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, 1)
        )

        self.layers_info = {"net": self.net, "fc_adv": self.fc_adv, "fc_val": self.fc_val}

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
        net_out = self.net(x)
        val = self.fc_val(net_out)
        adv = self.fc_adv(net_out)
        return val + adv - adv.mean()