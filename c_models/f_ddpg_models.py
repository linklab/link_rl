from typing import Tuple

from c_models.c_actor_models import DiscreteActorModel, ContinuousDeterministicActorModel
from c_models.d_critic_models import QCriticModel


# class DiscreteDdpgModel:
#     def __init__(
#             self, observation_shape: Tuple[int], n_out_actions: int, n_discrete_actions=None, config=None
#     ):
#         self.config = config
#
#         self.actor_model = DiscreteActorModel(
#             observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
#             config=self.config
#         ).to(self.config.DEVICE)
#
#         self.critic_model = QCriticModel(
#             observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=n_discrete_actions,
#             config=self.config
#         ).to(self.config.DEVICE)


class ContinuousDdpgModel:
    def __init__(
            self, observation_shape: Tuple[int], n_out_actions: int, config=None
    ):
        self.config = config

        self.actor_model = ContinuousDeterministicActorModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, config=self.config
        ).to(self.config.DEVICE)

        self.critic_model = QCriticModel(
            observation_shape=observation_shape, n_out_actions=n_out_actions, n_discrete_actions=None,
            config=self.config
        ).to(self.config.DEVICE)
