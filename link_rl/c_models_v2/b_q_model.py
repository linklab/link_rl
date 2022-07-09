import enum

import torch
from torch import nn
from typing import final, Tuple, List, Union, cast

from link_rl.c_models_v2.a_model import SingleModel, model_registry


class Q_MODEL(enum.Enum):
    QModel = "QModel"
    DuelingQModel = "DuelingQModel"
    EncoderQModel = "EncoderQModel"
    NatureAtariQModel = "NatureAtariQModel"


@model_registry.add
class QModel(SingleModel):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super().__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> nn.Module:
        model = nn.Sequential(
            nn.Linear(self._n_input, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, self._n_discrete_actions)
        )
        return model


@model_registry.add
class DuelingQModel(SingleModel):
    class DuelingQModel(nn.Module):
        def __init__(self, shared_net, adv_net, val_net):
            super().__init__()
            self.shared_net = shared_net
            self.adv_net = adv_net
            self.val_net = val_net

        def forward(self, obs):
            x = self.shared_net(obs)
            adv = self.adv_net(x)
            val = self.val_net(x)
            q_values = val + adv - torch.mean(adv, dim=-1, keepdim=True)

            return q_values

    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super().__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> nn.Module:
        shared_net = nn.Sequential(
            nn.Linear(self._n_input, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU()
        )
        adv_net = nn.Linear(128, self._n_discrete_actions)
        val_net = nn.Linear(128, 1)

        dueling_q_model = DuelingQModel.DuelingQModel(shared_net, adv_net, val_net)
        return dueling_q_model


@model_registry.add
class EncoderQModel(SingleModel):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super().__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> nn.Module:
        def make_vgg_layers(in_channels: int, config: List[Union[str, int]]) -> nn.Sequential:
            layers: List[nn.Module] = []
            _in_channels = in_channels
            for v in config:
                if v == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    channels = cast(int, v)
                    conv2d = nn.Conv2d(
                        in_channels=_in_channels,
                        out_channels=channels,
                        kernel_size=3,
                        padding=1
                    )
                    layers += [conv2d, nn.ReLU(inplace=True)]
                    _in_channels = channels
            return nn.Sequential(*layers)

        vgg16_layers = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        vgg16_net = make_vgg_layers(
            in_channels=self._n_input,
            config=vgg16_layers
        )

        encoder_out = self._get_encoder_out(
            conv_layers=vgg16_net,
            shape=self._observation_shape
        )

        linear_model = nn.Sequential(
            nn.Linear(encoder_out, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, self._n_discrete_actions)
        )

        model = nn.Sequential(
            vgg16_net,
            nn.Flatten(start_dim=1),
            linear_model
        )

        return model


@model_registry.add
class NatureAtariQModel(SingleModel):
    def __init__(
        self,
        n_input: int,
        n_out_actions: int,
        n_discrete_actions=None
    ):
        super().__init__(
            n_input,
            n_out_actions,
            n_discrete_actions
        )

    @final
    def _create_model(self) -> nn.Module:
        linear_model = nn.Sequential(
            nn.Linear(self._n_input, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self._n_discrete_actions)
        )

        model = nn.Sequential(
            linear_model
        )
        return model