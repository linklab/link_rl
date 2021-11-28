import torch.optim as optim
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.multiprocessing as mp

from c_models.models import ActorCritic
from d_agents.agent import Agent
from g_utils.buffers import Buffer
from g_utils.types import AgentMode


class AgentA2c(Agent):
    def __init__(self, n_features, n_actions, device, params):
        super(AgentA2c, self).__init__(
            n_features, n_actions, device, params
        )

        self.actor_critic_model = ActorCritic(
            n_features=n_features, n_actions=n_actions, device=device
        ).to(device)
        self.actor_critic_model.share_memory()

        self.optimizer = optim.Adam(
            self.actor_critic_model.parameters(), lr=self.params.LEARNING_RATE
        )

        self.model = self.actor_critic_model

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_log_actor_objective = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        action_prob = self.actor_critic_model.pi(obs)
        m = Categorical(probs=action_prob)
        if mode == AgentMode.TRAIN:
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=-1)
        return action.cpu().numpy()

    def train_a2c(self, buffer):
        # observations.shape: torch.Size([32, 4, 84, 84]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4, 84, 84]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])

        observations, actions, next_observations, rewards, dones = \
            buffer.sample(batch_size=self.params.BATCH_SIZE, device=self.device)

        self.optimizer.zero_grad()

        ###################################
        #  Critic (Value) 손실 산출 - BEGIN #
        ###################################
        # next_values.shape: (32, 1)
        next_values = self.actor_critic_model.v(next_observations)
        td_target_value_lst = []

        for reward, next_value, done in zip(rewards, next_values, dones):
            td_target = reward + self.params.GAMMA * next_value * (0.0 if done else 1.0)
            td_target_value_lst.append(td_target)

        # td_target_values.shape: (32, 1)
        td_target_values = torch.tensor(
            td_target_value_lst, dtype=torch.float32, device=self.device
        ).unsqueeze(dim=-1)

        # values.shape: (32, 1)
        values = self.actor_critic_model.v(observations)
        # loss_critic.shape: (,) <--  값 1개
        critic_loss = F.mse_loss(td_target_values.detach(), values)
        ###################################
        #  Critic (Value)  Loss 산출 - END #
        ###################################

        ################################
        #  Actor Objective 산출 - BEGIN #
        ################################
        q_values = td_target_values
        advantages = (q_values - values).detach()

        action_probs = self.actor_critic_model.pi(observations)
        # print(action_probs.shape, actions.unsqueeze(-1).shape, "!!!!!!!!!1")
        action_prob_selected = action_probs.gather(dim=1, index=actions)

        # action_prob_selected.shape: (32, 1)
        # advantage.shape: (32, 1)
        # log_pi_advantages.shape: (32, 1)
        log_pi_advantages = torch.multiply(
            torch.log(action_prob_selected), advantages
        )

        # actor_objective.shape: (,) <--  값 1개
        log_actor_objective = torch.sum(log_pi_advantages)

        actor_loss = torch.multiply(log_actor_objective, -1.0)
        ##############################
        #  Actor Objective 산출 - END #
        ##############################

        loss = critic_loss * 0.5 + actor_loss

        loss.backward()
        self.optimizer.step()

        self.last_critic_loss.value = critic_loss.item()
        self.last_log_actor_objective.value = log_actor_objective.item()
