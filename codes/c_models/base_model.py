# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import glob
import torch
import torch.nn as nn
import sys, os

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)


class BaseModel(nn.Module):
    def __init__(self, worker_id, params, device):
        super(BaseModel, self).__init__()

        self.__name__ = "BaseModel"

        self.worker_id = worker_id
        self.params = params
        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.base = None

        self.avg_gradients = {}
        self.weighted_scores = [0, 0, 0, 0]
        self.ema_scores = [0, 0, 0, 0]
        self.id_list = []
        self.weighted_gradients = {}
        self.count = 0
        self.sum = 0
        self.device = device

        self.steps_done = 0

        files = glob.glob(os.path.join(PROJECT_HOME, "out", "model_save_files", "{0}_{1}_{2}_*".format(
            self.worker_id,
            self.params.ENVIRONMENT_ID.name,
            self.params.DEEP_LEARNING_MODEL.value,
        )))

        if self.worker_id >= 0:
            if len(files) > 1:
                print("Worker ID - {0}: Problem occurs since there are two or more save files".format(self.worker_id))
            elif len(files) == 1:
                filename = files[0]
                self.load_state_dict(torch.load(filename))
                self.eval()
                print("Worker ID - {0}: Successful Model Load From {1}".format(self.worker_id, filename))
            else:
                print("Worker ID - {0}: There is no saved model".format(self.worker_id))

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)
        return self.base.forward(inputs)

    def reset_average_gradients(self):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            self.avg_gradients[layer_name] = {}
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] = torch.zeros(size=param.size()).to(self.device)

    def reset_weighted_gradients(self):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            self.weighted_gradients[layer_name] = {}
            for name, param in named_parameters:
                self.weighted_gradients[layer_name][name] = torch.zeros(size=param.size()).to(self.device)

    def get_gradients_for_current_parameters(self):
        gradients = {}

        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            gradients[layer_name] = {}
            for name, param in named_parameters:
                gradients[layer_name][name] = param.grad

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            for name, param in named_parameters:
                param.grad = gradients[layer_name][name]

    def accumulate_gradients(self, gradients):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] += gradients[layer_name][name]

    def update_average_gradients(self, num_workers):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] /= num_workers

    def update_score_weighted_gradients(self, num_workers, scores, gradients, worker_id, episode):
        self.ema_scores[worker_id] = scores[worker_id][-1]
        self.sum += scores[worker_id][-1]
        self.id_list.append(worker_id)

        if episode == 0:
            for layer_name, layer in self.base.layers_info.items():
                named_parameters = layer.to(self.device).named_parameters()
                for name, param in named_parameters:
                    self.avg_gradients[layer_name][name] += gradients[layer_name][name]
        else:
            for layer_name, layer in self.base.layers_info.items():
                named_parameters = layer.to(self.device).named_parameters()
                for name, param in named_parameters:
                    self.weighted_gradients[layer_name][name] += self.weighted_scores[self.id_list[self.count]] * gradients[layer_name][name]

        self.count += 1
        if self.count == num_workers:
            self.weighted_scores = [episode_reward / self.sum for episode_reward in self.ema_scores]
            self.count = 0
            self.sum = 0
            self.id_list = []

    def get_parameters(self):
        parameters = {}

        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            parameters[layer_name] = {}
            for name, param in named_parameters:
                parameters[layer_name][name] = param.data

        return parameters

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.to(self.device).named_parameters()
            for name, param in named_parameters:
                if soft_transfer:
                    param.data = param.data * (1.0 - soft_transfer_tau) + parameters[layer_name][name] * soft_transfer_tau
                else:
                    param.data = parameters[layer_name][name]

    def check_gradient_nan_or_zero(self, gradients):
        for layer_name, layer in gradients.items():
            for name, gradients in layer.items():
                if torch.unique(gradients).shape[0] == 1 and torch.sum(gradients).item() == 0.0:
                    print(layer_name, name, gradients)
                if torch.isnan(gradients).any():
                    print(layer_name, name, gradients)
                    raise ValueError()