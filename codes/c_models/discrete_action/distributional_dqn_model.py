import torch.nn as nn

from codes.c_models.discrete_action.dqn_model import DuelingDQNModel, DuelingDQN_MLP_Base, DuelingDQN_CNN_Base, \
    DuelingDQN_SmallCNN_Base


class DistDuelingDQNModel(DuelingDQNModel):
    def __init__(self, worker_id, input_shape=None, num_outputs=None, params=None, device=None):
        super(DistDuelingDQNModel, self).__init__(worker_id, params, device)
        self.__name__ = "DistDuelingDQNModel"


class DistDuelingDQN_MLP_Base(DuelingDQN_MLP_Base):
    def __init__(self, num_inputs, num_outputs, num_supports, params):
        super(DistDuelingDQN_MLP_Base, self).__init__(num_inputs, num_outputs, params)
        self.__name__ = "DistDuelingDQN_MLP_Base"

        self.dist_net_list = []
        for i in range(self.num_outputs):
            self.dist_net_list.append(nn.Linear(self.hidden_3_size, num_supports))


    def forward(self, x):
        q_value = super(DistDuelingDQN_MLP_Base, self).forward(self, x)
        distrinutions = []
        for i in range(self.num_outputs):
            distrinutions.append(self.dist_net_list[i](q_value))




class DistDuelingDQN_CNN_Base(DuelingDQN_CNN_Base):
    def __init__(self, input_shape, num_outputs, params):
        super(DistDuelingDQN_CNN_Base, self).__init__(input_shape, num_outputs, params)
        self.__name__ = "DistDuelingDQN_MLP_Base"


class DistDuelingDQN_SmallCNN_Base(DuelingDQN_SmallCNN_Base):
    def __init__(self, input_shape, num_outputs, params):
        super(DistDuelingDQN_SmallCNN_Base, self).__init__(input_shape, num_outputs, params)
        self.__name__ = "DistDuelingDQN_SmallCNN_Base"