import torch.optim as optim
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp

from link_rl.d_agents.on_policy.on_policy_agent import OnPolicyAgent


class AgentA2c(OnPolicyAgent):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentA2c, self).__init__(observation_space, action_space, config, need_train)

        # if isinstance(self.action_space, Discrete):
        #     self._model_creator = DiscreteBasicActorCriticModel(
        #         n_input=self.observation_shape[0],
        #         n_out_actions=self.n_out_actions,
        #         n_discrete_actions=self.n_discrete_actions
        #     )
        # elif isinstance(self.action_space, Box):
        #     self._model_creator = ContinuousBasicActorCriticModel(
        #         n_input=self.observation_shape[0],
        #         n_out_actions=self.n_out_actions,
        #         n_discrete_actions=self.n_discrete_actions
        #     )
        # else:
        #     raise ValueError()

        model = self._model_creator.create_model()
        self.actor_model, self.critic_model = model

        self.actor_model.to(self.config.DEVICE)
        self.critic_model.to(self.config.DEVICE)

        self.model = self.actor_model
        self.model.eval()

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)
        self.last_entropy = mp.Value('d', 0.0)

    def get_critic_loss(self, values, detached_target_values):
        assert values.shape == detached_target_values.shape, "{0} {1}".format(values.shape, detached_target_values.shape)
        critic_loss = self.config.LOSS_FUNCTION(values, detached_target_values.detach())
        return critic_loss

    def get_actor_loss(self, detached_advantages):
        if isinstance(self.action_space, Discrete):
            # action_probs.shape: (32, 2)
            # actions.shape: (32, 1)
            # advantage.shape: (32, 1)
            # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
            # criticized_log_pi_action_v.shape: (32,)

            action_probs = self.actor_model(self.observations)
            dist = Categorical(probs=action_probs)
            criticized_log_pi_action_v = dist.log_prob(value=self.actions.squeeze(dim=-1)) * detached_advantages.squeeze(dim=-1)

        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_model(self.observations)

            dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
            criticized_log_pi_action_v = dist.log_prob(value=self.actions).sum(dim=-1) * detached_advantages.squeeze(dim=-1)
        else:
            raise ValueError()

        entropy = torch.mean(dist.entropy())

        # actor_objective.shape: (,) <--  값 1개
        actor_objective = torch.mean(criticized_log_pi_action_v)
        actor_loss = -1.0 * actor_objective
        entropy_loss = -1.0 * entropy
        actor_loss = actor_loss + entropy_loss * self.config.ENTROPY_BETA

        return actor_loss, actor_objective, entropy

    def train_a2c(self):
        count_training_steps = 0

        # values.shape: (256, 1)
        # detached_target_values.shape: (256, 1)
        # detached_advantages.shape: (256, 1)
        values, detached_target_values, detached_advantages = self.get_target_values_and_advantages()
        #############################################
        #  Critic (Value) Loss 산출 & Update - BEGIN #
        #############################################
        critic_loss = self.get_critic_loss(values=values, detached_target_values=detached_target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.parameters())
        self.critic_optimizer.step()
        ##########################################
        #  Critic (Value) Loss 산출 & Update- END #
        ##########################################

        #########################################
        #  Actor Objective 산출 & Update - BEGIN #
        #########################################
        actor_loss, actor_objective, entropy = self.get_actor_loss(detached_advantages=detached_advantages)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.clip_actor_model_parameter_grad_value(self.actor_model.parameters())
        self.actor_optimizer.step()
        #######################################
        #  Actor Objective 산출 & Update - END #
        #######################################

        self.last_critic_loss.value = critic_loss.item()
        self.last_actor_objective.value = actor_objective.item()
        self.last_entropy.value = entropy.item()

        count_training_steps += 1

        return count_training_steps
