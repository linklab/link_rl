import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

# https://paperswithcode.com/paper/noisy-networks-for-exploration
# The register_buffer method creates a tensor in the network that won't be updated during backpropagation,
# but will be handled by the nn.Module machinery.
# For example, it will be copied to GPU with the cuda() call
# 1. optimizer가 업데이트하지 않는다.
# 2. 그러나 값은 존재한다
# 3. state_dict()로 확인이 가능하다.
# 4. GPU연산이 가능하다.


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # https://stackoverflow.com/questions/48869836/why-unintialized-tensor-in-pytorch-have-initial-values
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(dim=1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(dim=1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(dim=0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # self.weight_epsilon.size(): [2, 128] or [128, 128]
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))  # outer product

        # self.bias_epsilon.size(): [2] or [128]
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    # def sample_noise(self):
    #     self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    #     self.epsilon_bias = torch.randn(self.out_features)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)

        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )

        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))

        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init)
            )

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()

        noise_v = torch.mul(eps_in, eps_out)
        weight = self.weight + self.sigma_weight * noise_v

        return F.linear(input, weight, bias)


if __name__ == "__main__":
    noisy_net = NoisyLinear(3, 2)
    print(noisy_net.weight, noisy_net.bias)

    print("#### - 1")
    print(noisy_net.sigma_weight, noisy_net.sigma_bias)
    print(noisy_net.epsilon_weight, noisy_net.epsilon_bias)
    print("#### - 1")

    input_data = torch.Tensor([1, 1, 1])
    out = noisy_net(input_data)
    print(out)

    loss = F.l1_loss(out, torch.Tensor([0, 0]))
    optimizer = torch.optim.Adam(noisy_net.parameters(), lr=0.01)  # Define optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("#### - 2")
    print(noisy_net.sigma_weight, noisy_net.sigma_bias)
    print(noisy_net.epsilon_weight, noisy_net.epsilon_bias)
    print("#### - 2")