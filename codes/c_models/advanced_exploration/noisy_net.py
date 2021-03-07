import math

import torch
from torch import nn
from torch.nn import functional as F

# The register_buffer method creates a tensor in the network that won't be updated during backpropagation,
# but will be handled by the nn.Module machinery.
# For example, it will be copied to GPU with the cuda() call


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )

        self.register_buffer(
            "epsilon_weight", torch.zeros(out_features, in_features)
        )

        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init)
            )

            self.register_buffer(
                "epsilon_bias", torch.zeros(out_features)
            )

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        self.epsilon_weight.normal_()
        weight = self.weight + self.sigma_weight * self.epsilon_weight.data

        return F.linear(input, weight, bias)


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