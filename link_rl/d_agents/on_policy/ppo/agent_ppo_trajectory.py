import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal


from link_rl.d_agents.on_policy.ppo.agent_ppo import AgentPpo


class AgentPpoTrajectory(AgentPpo):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentPpoTrajectory, self).__init__(observation_space, action_space, config, need_train)

    def train_ppo(self):
        count_training_steps = 0

        #####################################
        # OLD_LOG_PI 얻어오기: BEGIN
        #####################################
        if isinstance(self.action_space, Discrete):
            trajectory_action_probs = self.actor_model.pi(self.observations)
            trajectory_dist = Categorical(probs=trajectory_action_probs)
            trajectory_old_log_pi_action_v = trajectory_dist.log_prob(value=self.actions.squeeze(dim=-1))
        elif isinstance(self.action_space, Box):
            trajectory_mu_v, trajectory_var_v = self.actor_model.pi(self.observations)
            trajectory_dist = Normal(loc=trajectory_mu_v, scale=torch.sqrt(trajectory_var_v))
            trajectory_old_log_pi_action_v = trajectory_dist.log_prob(value=self.actions).sum(dim=-1)
        else:
            raise ValueError()
        #####################################
        # OLD_LOG_PI 얻어오기: END
        # action_space가 Discrete와 Box에 관계없이 --> trajectory_old_log_pi_action_v.shape: (1280,)
        #####################################

        sum_critic_loss = 0.0
        sum_actor_objective = 0.0
        sum_ratio = 0.0
        sum_entropy = 0.0

        _, trajectory_detached_target_values, trajectory_detached_advantages = self.get_target_values_and_advantages()
        trajectory_detached_advantages = trajectory_detached_advantages.squeeze(dim=-1)  # NOTE

        for _ in range(self.config.PPO_K_EPOCH):
            for batch_offset in range(0, self.config.PPO_TRAJECTORY_SIZE, self.config.BATCH_SIZE):
                batch_l = batch_offset + self.config.BATCH_SIZE

                # Batch 단위의 훈련에 필요한 재료 마련
                batch_observations = self.observations[batch_offset:batch_l]
                batch_actions = self.actions[batch_offset:batch_l]
                batch_detached_target_values = trajectory_detached_target_values[batch_offset:batch_l]
                batch_old_log_pi_action_v = trajectory_old_log_pi_action_v[batch_offset:batch_l]
                batch_detached_advantages = trajectory_detached_advantages[batch_offset:batch_l]

                #############################################
                #  Critic (Value) Loss 산출 & Update - BEGIN #
                #############################################
                batch_values = self.critic_model.v(batch_observations)
                batch_critic_loss = self.train_critic(
                    values=batch_values, detached_target_values=batch_detached_target_values
                )
                ##########################################
                #  Critic (Value) Loss 산출 & Update- END #
                ##########################################

                #########################################
                #  Actor Objective 산출 & Update - BEGIN #
                #########################################
                if isinstance(self.action_space, Discrete):
                    # actions.shape: (32, 1)
                    # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
                    # criticized_log_pi_action_v.shape: (32,)
                    batch_action_probs = self.actor_model.pi(batch_observations)
                    batch_dist = Categorical(probs=batch_action_probs)
                    batch_log_pi_action_v = batch_dist.log_prob(value=batch_actions.squeeze(dim=-1))
                    batch_entropy = batch_dist.entropy().mean()
                elif isinstance(self.action_space, Box):
                    batch_mu_v, batch_var_v = self.actor_model.pi(batch_observations)

                    # batch_log_pi_action_v = self.calc_log_prob(batch_mu_v, batch_var_v, batch_actions)
                    # batch_entropy = 0.5 * (torch.log(2.0 * np.pi * batch_var_v) + 1.0).sum(dim=-1)
                    # batch_entropy = batch_entropy.mean()

                    batch_dist = Normal(loc=batch_mu_v, scale=torch.sqrt(batch_var_v))
                    batch_log_pi_action_v = batch_dist.log_prob(value=batch_actions).sum(dim=-1)
                    batch_entropy = batch_dist.entropy().mean()

                    #print(batch_mu_v.shape, batch_var_v.shape, batch_actions.shape, batch_log_pi_action_v.shape, batch_entropy.shape, "@@@")
                else:
                    raise ValueError()

                batch_ratio = torch.exp(batch_log_pi_action_v - batch_old_log_pi_action_v.detach())

                assert batch_ratio.shape == batch_detached_advantages.shape, "{0}, {1}".format(
                    batch_ratio.shape, batch_detached_advantages.shape
                )
                batch_surrogate_objective_1 = batch_ratio * batch_detached_advantages
                batch_surrogate_objective_2 = torch.clamp(
                    batch_ratio, 1.0 - self.config.PPO_EPSILON_CLIP, 1.0 + self.config.PPO_EPSILON_CLIP
                ) * batch_detached_advantages

                assert batch_surrogate_objective_1.shape == batch_surrogate_objective_2.shape, "{0} {1}".format(
                    batch_surrogate_objective_1.shape, batch_surrogate_objective_2.shape
                )
                batch_actor_objective = torch.mean(torch.min(batch_surrogate_objective_1, batch_surrogate_objective_2))
                batch_actor_loss = -1.0 * batch_actor_objective
                batch_entropy_loss = -1.0 * batch_entropy
                batch_actor_loss = batch_actor_loss + batch_entropy_loss * self.config.ENTROPY_BETA

                self.actor_optimizer.zero_grad()
                batch_actor_loss.backward()
                self.clip_actor_model_parameter_grad_value(self.actor_model.parameters())
                self.actor_optimizer.step()

                sum_critic_loss += batch_critic_loss.item()
                sum_actor_objective += batch_actor_objective.item()
                sum_entropy += batch_entropy.item()
                sum_ratio += batch_ratio.mean().item()

                count_training_steps += 1
                #######################################
                #  Actor Objective 산출 & Update - END #
                #######################################

        self.synchronize_models(source_model=self.actor_model, target_model=self.actor_model)

        self.last_critic_loss.value = sum_critic_loss / count_training_steps
        self.last_actor_objective.value = sum_actor_objective / count_training_steps
        self.last_entropy.value = sum_entropy / count_training_steps
        self.last_ratio.value = sum_ratio / count_training_steps

        return count_training_steps
