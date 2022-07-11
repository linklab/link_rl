from link_rl.e_agents.on_policy.ppo.agent_ppo import AgentPpo


class AgentAsynchronousPpo(AgentPpo):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentAsynchronousPpo, self).__init__(observation_space, action_space, config, need_train)

    def train_asynchronous_ppo(self):
        pass


class WorkingAsynchronousPpo(AgentPpo):
    def __init__(self, master_agent, observation_space, action_space, shared_model_access_lock, config, need_train):
        super(WorkingAsynchronousPpo, self).__init__(observation_space, action_space, config, need_train)

        self.master_agent = master_agent
        self.shared_model_access_lock = shared_model_access_lock

    def worker_train(self):
        count_training_steps = 0

        old_log_pi_action_v = self.process_with_old_log_pi()

        sum_critic_loss = 0.0
        sum_actor_objective = 0.0
        sum_ratio = 0.0
        sum_entropy = 0.0

        _, detached_target_values, detached_advantages = self.get_target_values_and_advantages()
        detached_advantages = detached_advantages.squeeze(dim=-1)  # NOTE

        for _ in range(self.config.PPO_K_EPOCH):
            #############################################
            #  Critic (Value) Loss 산출 & Update - BEGIN #
            #############################################
            values = self.critic_model.v(self.observations)
            critic_loss = self.get_critic_loss(values=values, detached_target_values=detached_target_values)

            # calculate local gradients and push local worker parameters to master parameters
            self.master_agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            for master_critic_model_parameter, worker_critic_model_parameter in zip(
                    self.master_agent.critic_model.parameters(), self.critic_model.parameters()
            ):
                master_critic_model_parameter._grad = worker_critic_model_parameter.grad
            self.master_agent.clip_critic_model_parameter_grad_value(self.master_agent.critic_model.parameters())
            self.master_agent.critic_optimizer.step()

            # pull global parameters
            self.synchronize_models(source_model=self.master_agent.critic_model, target_model=self.critic_model)

            if self.encoder_is_not_identity:
                self.train_encoder()
            ##########################################
            #  Critic (Value) Loss 산출 & Update- END #
            ##########################################

            #########################################
            #  Actor Objective 산출 & Update - BEGIN #
            #########################################
            actor_loss, actor_objective, entropy, ratio = self.get_ppo_actor_loss(
                old_log_pi_action_v=old_log_pi_action_v, detached_advantages=detached_advantages
            )

            # calculate local gradients and push local worker parameters to master parameters
            self.master_agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            for master_actor_model_parameter, worker_actor_model_parameter in zip(
                    self.master_agent.actor_model.parameters(), self.actor_model.parameters()
            ):
                master_actor_model_parameter._grad = worker_actor_model_parameter.grad
            self.master_agent.clip_actor_model_parameter_grad_value(self.master_agent.actor_model.parameters())
            self.master_agent.actor_optimizer.step()

            # pull global parameters
            self.synchronize_models(source_model=self.master_agent.actor_model, target_model=self.actor_model)

            sum_critic_loss += critic_loss.item()
            sum_actor_objective += actor_objective.item()
            sum_entropy += entropy.item()
            sum_ratio += ratio.mean().item()

            count_training_steps += 1
            #######################################
            #  Actor Objective 산출 & Update - END #
            #######################################

        self.master_agent.last_critic_loss.value = sum_critic_loss / count_training_steps
        self.master_agent.last_actor_objective.value = sum_actor_objective / count_training_steps
        self.master_agent.last_entropy.value = sum_entropy / count_training_steps
        self.master_agent.last_ratio.value = sum_ratio / count_training_steps

        return count_training_steps
