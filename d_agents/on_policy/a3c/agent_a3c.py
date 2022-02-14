import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal

from d_agents.on_policy.a2c.agent_a2c import AgentA2c


class AgentA3c(AgentA2c):
    def __init__(self, observation_space, action_space, config):
        super(AgentA3c, self).__init__(observation_space, action_space, config)

    def train_a3c(self):
        pass


class WorkerAgentA3c(AgentA2c):
    def __init__(self, master_agent, observation_space, action_space, shared_model_access_lock, config):
        super(WorkerAgentA3c, self).__init__(observation_space, action_space, config)
        self.master_agent = master_agent

        self.critic_optimizer = self.master_agent.critic_optimizer
        self.actor_optimizer = self.master_agent.actor_optimizer

        self.shared_model_access_lock = shared_model_access_lock

    def worker_train(self):
        self._before_train(sample_length=self.config.BATCH_SIZE)

        count_training_steps = 0

        self.shared_model_access_lock.acquire()

        #############################################
        #  Critic (Value) Loss 산출 & Update - BEGIN #
        #############################################
        target_values = self.get_target_values(self.next_observations, self.rewards, self.dones)

        # values.shape: (32, 1)
        values = self.critic_model.v(self.observations)

        critic_loss = self.config.LOSS_FUNCTION(values, target_values.detach())

        # calculate local gradients and push local worker parameters to master parameters
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        for master_parameter, worker_parameter in zip(
                self.master_agent.critic_model.parameters(), self.critic_model.parameters()
        ):
            master_parameter._grad = worker_parameter.grad
        self.clip_critic_model_parameter_grad_value(self.master_agent.critic_model.critic_params_list)
        self.critic_optimizer.step()

        # pull global parameters
        self.synchronize_models(source_model=self.master_agent.critic_model, target_model=self.critic_model)
        ##########################################
        #  Critic (Value) Loss 산출 & Update- END #
        ##########################################

        #########################################
        #  Actor Objective 산출 & Update - BEGIN #
        #########################################
        advantages = (target_values - values).detach()
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-7)

        if isinstance(self.action_space, Discrete):
            action_probs = self.actor_model.pi(self.observations)
            dist = Categorical(probs=action_probs)

            # action_probs.shape: (32, 2)
            # actions.shape: (32, 1)
            # advantage.shape: (32, 1)
            # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
            # criticized_log_pi_action_v.shape: (32,)
            criticized_log_pi_action_v = dist.log_prob(value=self.actions.squeeze(dim=-1)) * advantages.squeeze(dim=-1)

            entropy = dist.entropy().mean()
        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_model.pi(self.observations)

            # criticized_log_pi_action_v = self.calc_log_prob(mu_v, var_v, self.actions) * advantages
            # entropy = 0.5 * (torch.log(2.0 * np.pi * var_v) + 1.0).sum(dim=-1)
            # entropy = entropy.mean()

            dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
            criticized_log_pi_action_v = dist.log_prob(value=self.actions).sum(dim=-1) * advantages.squeeze(dim=-1)
            entropy = dist.entropy().mean()
        else:
            raise ValueError()

        # actor_objective.shape: (,) <--  값 1개
        actor_objective = torch.mean(criticized_log_pi_action_v)

        actor_loss = -1.0 * actor_objective
        entropy_loss = -1.0 * entropy
        actor_loss = actor_loss + entropy_loss * self.config.ENTROPY_BETA

        # calculate local gradients and push local worker parameters to master parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for master_parameter, worker_parameter in zip(
                self.master_agent.actor_model.parameters(), self.actor_model.parameters()
        ):
            master_parameter._grad = worker_parameter.grad
        self.clip_actor_model_parameter_grad_value(self.master_agent.actor_model.actor_params_list)
        self.actor_optimizer.step()

        # pull global parameters
        self.synchronize_models(source_model=self.master_agent.actor_model, target_model=self.actor_model)
        #######################################
        #  Actor Objective 산출 & Update - END #
        #######################################

        self.shared_model_access_lock.release()

        self.master_agent.last_critic_loss.value = critic_loss.item()
        self.master_agent.last_actor_objective.value = actor_objective.item()
        self.master_agent.last_entropy.value = entropy.item()

        count_training_steps += 1

        self.buffer.clear()                 # ON_POLICY!
        self._after_train()

        return count_training_steps
