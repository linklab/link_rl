# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import glob
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from config.names import PROJECT_HOME, RLAlgorithmName


class ActorCriticModel(nn.Module):
    def __init__(self, s_size, a_size, worker_id, params, device, rl_algorithm=RLAlgorithmName.DDPG_FAST_V0):
        super(ActorCriticModel, self).__init__()

        self.__name__ = "ActorCriticModel"

        self.worker_id = worker_id
        self.params = params
        self.a_size = a_size
        self.rl_algorithm = rl_algorithm

        if self.rl_algorithm in [RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.DDPG_FAST_DOUBLE_AGENTS_V0]:
            self.base = DDPGActorCriticMLPBase(
                num_inputs=s_size,
                num_ouputs=a_size,
                params=self.params
            )
        elif self.rl_algorithm == RLAlgorithmName.D4PG_FAST_V0:
            self.base = D4PGActorCriticMLPBase(
                num_inputs=s_size,
                num_ouputs=a_size,
                params=self.params
            )
        else:
            raise ValueError()

        self.avg_gradients = {}
        self.weighted_scores = [0, 0, 0, 0]
        self.ema_scores = [0, 0, 0, 0]
        self.id_list = []
        self.weighted_gradients = {}
        self.count = 0
        self.sum = 0
        self.device = device

        self.reset_average_gradients()

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
        return self.base.forward_actor(inputs)

    def act(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)
        actions = self.base.forward_actor(inputs)
        return actions, None

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


class DDPGActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_ouputs, params):
        super(DDPGActorCriticMLPBase, self).__init__()
        self.__name__ = "DDPGActorCriticMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, num_ouputs),
        )

        self.actor.apply(self.init_weights)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs + num_ouputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, 1)
        )

        self.critic.apply(self.init_weights)

        self.layers_info = {'actor': self.actor, 'critic': self.critic}

        self.train()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        return self.forward_actor(inputs)

    def forward_actor(self, inputs):
        actions = self.actor(inputs)
        actions = torch.tanh(actions)

        return actions * self.params.ACTION_SCALE

    def forward_critic(self, inputs, actions):
        critic_value = self.critic(torch.cat([inputs, actions], dim=1))

        return critic_value


class D4PGActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_ouputs, params):
        super(D4PGActorCriticMLPBase, self).__init__()
        self.__name__ = "D4PGActorCriticMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, num_ouputs),
        )

        self.logstd = nn.Parameter(torch.zeros(num_ouputs))

        self.actor.apply(self.init_weights)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs + num_ouputs, self.hidden_1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.ReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.ReLU(),
            nn.Linear(self.hidden_3_size, params.N_ATOMS)
        )

        self.critic.apply(self.init_weights)

        self.layers_info = {'actor': self.actor, 'critic': self.critic}

        delta = (params.V_MAX - params.V_MIN) / (params.N_ATOMS - 1)
        self.register_buffer("supports", torch.arange(params.V_MIN, params.V_MAX + delta, delta))

        self.train()

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        return self.forward_actor(inputs)

    def forward_actor(self, inputs):
        actions = self.actor(inputs)
        actions = torch.tanh(actions)

        return actions * self.params.ACTION_SCALE

    def forward_critic(self, inputs, actions):
        critic_value = self.critic(torch.cat([inputs, actions], dim=1))

        return critic_value

    def distribution_to_q(self, distribution):
        weights = F.softmax(distribution, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
