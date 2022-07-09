from link_rl.e_agents.on_policy.a2c.agent_a2c import AgentA2c


class AgentA3c(AgentA2c):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentA3c, self).__init__(observation_space, action_space, config, need_train)

    def train_a3c(self):
        pass


class WorkingAgentA3c(AgentA2c):
    def __init__(self, master_agent, observation_space, action_space, shared_model_access_lock, config, need_train):
        super(WorkingAgentA3c, self).__init__(observation_space, action_space, config, need_train)

        self.master_agent = master_agent
        self.shared_model_access_lock = shared_model_access_lock

    def worker_train(self):
        count_training_steps = 0

        values, detached_target_values, detached_advantages = self.get_target_values_and_advantages()
        #############################################
        #  Critic (Value) Loss 산출 & Update - BEGIN #
        #############################################
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
        actor_loss, actor_objective, entropy = self.get_actor_loss(detached_advantages=detached_advantages)

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
        #######################################
        #  Actor Objective 산출 & Update - END #
        #######################################

        self.master_agent.last_critic_loss.value = critic_loss.item()
        self.master_agent.last_actor_objective.value = actor_objective.item()
        self.master_agent.last_entropy.value = entropy.item()

        count_training_steps += 1

        return count_training_steps