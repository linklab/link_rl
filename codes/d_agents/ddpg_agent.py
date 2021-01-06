import numpy as np
import torch
import torch.nn.functional as F

from codes.d_agents.a0_base_agent import BaseAgent, TargetNet, float32_preprocessor
from codes.e_utils import rl_utils, replay_buffer


class AgentDDPG(BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, num_inputs, num_outputs, action_min, action_max, worker_id, action_selector, params,
                 preprocessor=float32_preprocessor, device="cpu"):
        super(AgentDDPG, self).__init__()
        self.__name__ = "AgentDDPG"
        self.device = device
        self.preprocessor = preprocessor
        self.action_selector = action_selector
        self.action_min = action_min
        self.action_max = action_max
        self.step_idx = 0

        self.worker_id = worker_id
        self.params = params
        self.device = device

        self.model = rl_utils.get_rl_model(
            worker_id=worker_id, num_inputs=num_inputs, num_outputs=num_outputs, params=params, device=self.device
        )

        print(self.model.base.actor)
        print(self.model.base.critic)

        self.target_agent = TargetNet(self.model.base)

        self.actor_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.actor.parameters(),
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        self.critic_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.critic.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        if self.params.PER:
            self.buffer = replay_buffer.PrioritizedReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE,
                n_step=self.params.N_STEP, beta_start=0.4, beta_frames=self.params.MAX_GLOBAL_STEP
            )
        else:
            self.buffer = replay_buffer.ExperienceReplayBuffer(
                experience_source=None, buffer_size=self.params.REPLAY_BUFFER_SIZE
            )

    def __call__(self, states, agent_states=None):
        if self.preprocessor:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        mu_v = self.model(states)
        mu = mu_v.data.cpu().numpy()

        ####################################
        # if agent_states is None:
        #     new_agent_states = [None] * len(states)
        # else:
        #     new_agent_states = agent_states
        #
        # noises_v = torch.Tensor(self.ou_noise.noise()).unsqueeze(dim=-1).to(self.device)
        # noises = noises_v.data.cpu().numpy()
        #
        # actions = mu + noises
        # actions = np.clip(actions, self.action_min, self.action_max)
        ####################################

        actions, new_agent_states = self.action_selector(mu, agent_states)
        actions = np.clip(actions, self.action_min, self.action_max)
        #####################################

        self.step_idx += 1

        return actions, new_agent_states

    def train_net(self, step_idx):
        if self.params.PER:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        batch_states_v, batch_actions_v, batch_rewards_v, batch_dones_mask, batch_last_states_v \
            = self.unpack_batch_for_ddpg(batch)

        # train critic
        self.critic_optimizer.zero_grad()
        critic_parameters = self.model.base.critic.parameters()
        for p in critic_parameters:
            p.requires_grad = True

        batch_q_v = self.model.base.forward_critic(batch_states_v, batch_actions_v)
        batch_last_act_v = self.target_agent.target_model.forward_actor(batch_last_states_v)
        batch_q_last_v = self.target_agent.target_model.forward_critic(batch_last_states_v, batch_last_act_v)
        batch_q_last_v[batch_dones_mask] = 0.0
        batch_target_q_v = batch_rewards_v.unsqueeze(dim=-1) + batch_q_last_v * self.params.GAMMA ** self.params.N_STEP

        if self.params.PER:
            batch_l1_loss = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach(), reduction='none')  # for PER
            batch_weights_v = torch.tensor(batch_weights)
            critic_loss_v = batch_weights_v * batch_l1_loss

            self.buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
            self.buffer.update_beta(step_idx)
        else:
            critic_loss_v = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach())

        loss_critic_v = critic_loss_v.mean()

        loss_critic_v.backward()
        self.critic_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()
        critic_parameters = self.model.base.critic.parameters()
        for p in critic_parameters:
            p.requires_grad = False

        batch_current_actions_v = self.model.base.forward_actor(batch_states_v)
        actor_loss_v = -1.0 * self.model.base.forward_critic(batch_states_v, batch_current_actions_v)
        loss_actor_v = actor_loss_v.mean()

        loss_actor_v.backward()

        self.actor_optimizer.step()

        self.target_agent.alpha_sync(alpha=1 - 0.001)

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0

    def unpack_batch_for_ddpg(self, batch):
        states, actions, rewards, dones, last_states = [], [], [], [], []

        for exp in batch:
            states.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(exp.state)   # the result will be masked anyway
            else:
                last_states.append(np.array(exp.last_state, copy=False))

        states_v = float32_preprocessor(states).to(self.device)
        actions_v = float32_preprocessor(actions).to(self.device)
        rewards_v = float32_preprocessor(rewards).to(self.device)
        last_states_v = float32_preprocessor(last_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        return states_v, actions_v, rewards_v, dones_t, last_states_v