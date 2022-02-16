import torch.optim as optim
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp

from c_models.e_a2c_models import ContinuousActorCriticModel, DiscreteActorCriticModel
from d_agents.agent import OnPolicyAgent


class AgentA2c(OnPolicyAgent):
    def __init__(self, observation_space, action_space, config):
        super(AgentA2c, self).__init__(observation_space, action_space, config)

        if isinstance(self.action_space, Discrete):
            self.actor_critic_model = DiscreteActorCriticModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, config=config
            )
        elif isinstance(self.action_space, Box):
            self.actor_critic_model = ContinuousActorCriticModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, config=config
            )
        else:
            raise ValueError()

        self.actor_model = self.actor_critic_model.actor_model
        self.critic_model = self.actor_critic_model.critic_model

        self.model = self.actor_model  # 에이전트 밖에서는 model이라는 이름으로 제어 모델 접근

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)
        self.last_entropy = mp.Value('d', 0.0)

    def train_critic(self, values, target_values):
        #############################################
        #  Critic (Value) Loss 산출 & Update - BEGIN #
        #############################################
        assert values.shape == target_values.shape, "{0} {1}".format(values.shape, target_values.shape)
        critic_loss = self.config.LOSS_FUNCTION(values, target_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params_list)
        self.critic_optimizer.step()
        ##########################################
        #  Critic (Value) Loss 산출 & Update- END #
        ##########################################

        return critic_loss

    def train_a2c(self):
        count_training_steps = 0

        target_values, advantages = self.get_target_values_and_advantages()

        #############################################
        #  Critic (Value) Loss 산출 & Update - BEGIN #
        #############################################
        # target_values = self.get_target_values()
        values = self.critic_model.v(self.observations)

        critic_loss = self.train_critic(values=values, target_values=target_values)
        ##########################################
        #  Critic (Value) Loss 산출 & Update- END #
        ##########################################

        #########################################
        #  Actor Objective 산출 & Update - BEGIN #
        #########################################
        if isinstance(self.action_space, Discrete):
            action_probs = self.actor_model.pi(self.observations)
            dist = Categorical(probs=action_probs)

            # action_probs.shape: (32, 2)
            # actions.shape: (32, 1)
            # advantage.shape: (32, 1)
            # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
            # criticized_log_pi_action_v.shape: (32,)
            criticized_log_pi_action_v = dist.log_prob(value=self.actions.squeeze(dim=-1)) * advantages.squeeze(dim=-1)

        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_model.pi(self.observations)

            dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
            criticized_log_pi_action_v = dist.log_prob(value=self.actions).sum(dim=-1) * advantages.squeeze(dim=-1)
        else:
            raise ValueError()

        entropy = dist.entropy().mean()

        # actor_objective.shape: (,) <--  값 1개
        actor_objective = torch.mean(criticized_log_pi_action_v)

        # if self.step % 1000 == 0:
        #     print("mu_v:", mu_v, "var_v:", var_v)
        #     print("actor_objective:", actor_objective)
        #     print("target_values:", target_values)
        #     print("values:", values)
        #     print("advantages:", advantages)
        #     print("self.actions:", self.actions)
        #     print("dist.log_prob(value=self.actions).sum(dim=-1, keepdim=True):", dist.log_prob(value=self.actions).sum(dim=-1, keepdim=True))

        actor_loss = -1.0 * actor_objective
        entropy_loss = -1.0 * entropy
        actor_loss = actor_loss + entropy_loss * self.config.ENTROPY_BETA

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params_list)
        self.actor_optimizer.step()
        #######################################
        #  Actor Objective 산출 & Update - END #
        #######################################

        self.last_critic_loss.value = critic_loss.item()
        self.last_actor_objective.value = actor_objective.item()
        self.last_entropy.value = entropy.item()

        count_training_steps += 1

        return count_training_steps
