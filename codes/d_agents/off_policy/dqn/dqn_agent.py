import numpy as np
import torch
import torch.nn.functional as F

from codes.b_environments.trade.trade_action_selector import EpsilonGreedyTradeDQNActionSelector, \
    ArgmaxTradeActionSelector
from codes.c_models.discrete_action.dqn_model import DuelingDQNModel
from codes.d_agents.a0_base_agent import TargetNet, float32_preprocessor
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.e_utils import rl_utils
from codes.e_utils.actions import EpsilonGreedyDQNActionSelector, ArgmaxActionSelector, EpsilonTracker, \
    EpsilonGreedySomeTimesBlowDQNActionSelector
from codes.e_utils.names import DeepLearningModelName, AgentMode, EnvironmentName


class AgentDQN(OffPolicyAgent):
    """
    """
    def __init__(self, worker_id, input_shape, action_shape, num_outputs, params, device):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.DUELING_DQN_MLP,
            DeepLearningModelName.DUELING_DQN_CNN,
            DeepLearningModelName.DUELING_DQN_SMALL_CNN
        ]

        super(AgentDQN, self).__init__(worker_id=worker_id, params=params, action_shape=action_shape, device=device)
        if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0]:
            self.train_action_selector = EpsilonGreedySomeTimesBlowDQNActionSelector(
                epsilon=params.EPSILON_INIT, blowing_action_rate=0.0002,
                min_blowing_action_idx=0, max_blowing_action_idx=-1
            )
            self.test_and_play_action_selector = EpsilonGreedySomeTimesBlowDQNActionSelector(
                epsilon=0.0, blowing_action_rate=0.0002, min_blowing_action_idx=0, max_blowing_action_idx=-1
            )
        elif params.ENVIRONMENT_ID in [EnvironmentName.TRADE_V0]:
            # main 에서 action_selector 할당
            self.train_action_selector = None
            self.test_and_play_action_selector = None
        else:
            if self.params.NOISY_NET:
                self.train_action_selector = ArgmaxActionSelector()
            else:
                self.train_action_selector = EpsilonGreedyDQNActionSelector(epsilon=params.EPSILON_INIT)
            self.test_and_play_action_selector = ArgmaxActionSelector()

        if self.params.NOISY_NET:
            self.epsilon_tracker = None
        else:
            self.epsilon_tracker = EpsilonTracker(
                action_selector=self.train_action_selector,
                eps_start=params.EPSILON_INIT,
                eps_final=params.EPSILON_MIN,
                eps_frames=params.EPSILON_MIN_STEP
            )

        self.__name__ = "AgentDQN"

        if self.params.DISTRIBUTIONAL:
            pass
        else:
            self.model = DuelingDQNModel(
                worker_id=worker_id,
                input_shape=input_shape,
                num_outputs=num_outputs,
                params=params,
                device=device
            ).to(device)

        self.target_agent = TargetNet(self.model.base)

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

    def __call__(self, states, agent_states=None):
        if not agent_states:
            agent_states = [None] * len(states)

        if not isinstance(states, torch.FloatTensor):
            states = float32_preprocessor(states).to(self.device)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.params.DISTRIBUTIONAL:
            actions = None
        else:
            q_v = self.model(states)
            q = q_v.detach().cpu().numpy()

            if self.agent_mode == AgentMode.TRAIN:
                actions = self.train_action_selector(q)
            else:
                actions = self.test_and_play_action_selector(q)

        return actions, agent_states

    def train(self, step_idx):
        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        self.optimizer.zero_grad()

        if self.params.OMEGA:
            assert self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED
            loss_v, sample_prios = self.calc_loss_per_double_dqn_for_omega(batch, batch_indices, batch_weights)
            self.buffer.update_priorities(batch_indices, sample_prios.detach().cpu().numpy())
            self.buffer.update_beta(step_idx)
        else:
            if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
                loss_v, sample_prios = self.calc_loss_per_double_dqn(batch, batch_indices, batch_weights)
                self.buffer.update_priorities(batch_indices, sample_prios.detach().cpu().numpy())
                self.buffer.update_beta(step_idx)
            else:
                if self.params.DOUBLE:
                    loss_v = self.calc_loss_double_dqn(batch)
                else:
                    loss_v = self.calc_loss_dqn(batch)

        loss_v.backward()

        self.optimizer.step()

        if step_idx % self.params.TARGET_NET_SYNC_STEP_PERIOD < self.params.TRAIN_STEP_FREQ:
            self.target_agent.sync()

        gradients = self.model.get_gradients_for_current_parameters()

        self.model.check_gradient_nan_or_zero(gradients)

        if self.params.NOISY_NET:
            self.model.base.reset_noise()  # Pick a new noise vector (until next optimisation step)
            self.target_agent.target_model.reset_noise()

        return gradients, loss_v.detach().item(), None

    def unpack_batch(self, batch):
        states, actions, rewards, dones, last_states, last_steps = [], [], [], [], [], []

        for exp in batch:
            state = np.array(exp.state, copy=False)
            states.append(state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            last_steps.append(exp.last_step)
            if exp.last_state is None:
                last_states.append(state)  # the result will be masked anyway
            else:
                last_states.append(np.array(exp.last_state, copy=False))
        return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(last_states, copy=False), np.array(last_steps)

    def unpack_batch_for_n_step(self, batch, batch_indices):
        states, actions, rewards, dones, next_states, last_steps = [], [], [], [], [], []
        for idx, exp in enumerate(batch):
            state = np.array(exp.state, copy=False)
            states.append(state)
            actions.append(exp.action)

            n_step_rewards = 0
            gamma = 1
            current_exp = exp
            for i in range(self.params.N_STEP):
                n_step_rewards += gamma * current_exp.reward
                next_exp = self.buffer.buffer[(batch_indices[idx] + i + 1) % self.params.REPLAY_BUFFER_SIZE]

                if current_exp.done:
                    rewards.append(n_step_rewards)
                    next_states.append(np.array(next_exp.state, copy=False))
                    dones.append(True)
                    last_steps.append(i + 1)
                    break
                else:
                    if i == self.params.N_STEP - 1:
                        rewards.append(n_step_rewards)
                        next_states.append(np.array(next_exp.state, copy=False))
                        dones.append(False)
                        last_steps.append(i + 1)

                current_exp = next_exp
                gamma *= self.params.GAMMA

        return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states, copy=False), np.array(last_steps)

    def unpack_batch_for_omega(self, batch, batch_indices):
        states, actions, rewards, done_mask, next_states = [], [], [], [], []
        for idx, exp in enumerate(batch):
            state = np.array(exp.state, copy=False)
            states.append(state)
            actions.append(exp.action)

            n_step_rewards = []
            current_exp = exp
            for i in range(self.params.OMEGA_WINDOW_SIZE):
                n_step_rewards.append(current_exp.reward)
                next_exp = self.buffer.buffer[(batch_indices[idx] + i + 1) % self.params.REPLAY_BUFFER_SIZE]
                next_states.append(np.array(next_exp.state, copy=False))

                if current_exp.done:
                    done_mask.append(0)
                    break
                else:
                    if i == self.params.OMEGA_WINDOW_SIZE - 1:
                        done_mask.append(1)

                current_exp = next_exp

            rewards.append(n_step_rewards)

        return np.array(states, copy=False), np.array(actions), np.array(rewards), np.array(done_mask), np.array(
            next_states, copy=False)

    def calc_loss_dqn(self, batch):
        states, actions, rewards, dones, next_states, last_steps = self.unpack_batch(batch)

        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        done_mask = torch.BoolTensor(dones)
        last_steps_v = torch.tensor(last_steps)
        if self.device == torch.device("cuda"):
            states_v = states_v.cuda(non_blocking=True)
            next_states_v = next_states_v.cuda(non_blocking=True)
            actions_v = actions_v.cuda(non_blocking=True)
            rewards_v = rewards_v.cuda(non_blocking=True)
            done_mask = done_mask.cuda(non_blocking=True)
            last_steps_v = last_steps_v.cuda(non_blocking=True)

        # https://subscription.packtpub.com/book/data/9781838826994/6/ch06lvl1sec45/dqn-on-pong
        # We pass observations to the first model and extract the specific Q - values for the taken actions using the gather() tensor operation.
        # The first argument to the gather() call is a dimension index that we want to perform gathering on.
        # In our case, it is equal to 1, which corresponds to actions.
        # The second argument is a tensor of indices of elements to be chosen.
        # Extra unsqueeze() and squeeze() calls are required to compute the index argument for the gather functions,
        # and to get rid of the extra dimensions that we created, respectively.
        # The index should have the same number of dimensions as the data we are processing.
        # In Figure 6.3, you can see an illustration of what gather() does on the example case, with a batch of six entries and four actions:
        action_values = self.model(states_v).gather(dim=1, index=actions_v.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_agent.target_model(next_states_v).max(1)[0]
            next_state_values[done_mask] = 0.0

        target_action_values = next_state_values.detach() * (self.params.GAMMA ** last_steps_v) + rewards_v

        # return nn.MSELoss()(action_values, target_action_values)
        return F.smooth_l1_loss(action_values, target_action_values)

    def calc_loss_double_dqn(self, batch):
        states, actions, rewards, dones, next_states, last_steps = self.unpack_batch(batch)

        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        done_mask = torch.BoolTensor(dones)
        last_steps_v = torch.tensor(last_steps)

        if self.device == torch.device("cuda"):
            states_v = states_v.cuda(non_blocking=True)
            next_states_v = next_states_v.cuda(non_blocking=True)
            actions_v = actions_v.cuda(non_blocking=True)
            rewards_v = rewards_v.cuda(non_blocking=True)
            done_mask = done_mask.cuda(non_blocking=True)
            last_steps_v = last_steps_v.cuda(non_blocking=True)

        actions_v = actions_v.unsqueeze(-1)
        action_values = self.model(states_v).gather(1, actions_v)
        action_values = action_values.squeeze(-1)
        with torch.no_grad():
            next_state_acts = self.model(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_vals = self.target_agent.target_model(next_states_v).gather(1, next_state_acts).squeeze(-1)
            next_state_vals[done_mask] = 0.0

        exp_sa_vals = next_state_vals.detach() * (self.params.GAMMA ** last_steps_v) + rewards_v

        # print(action_values, exp_sa_vals, "#####")

        # return nn.MSELoss()(action_values, exp_sa_vals)
        return F.smooth_l1_loss(action_values, exp_sa_vals)

    def calc_loss_per_double_dqn(self, batch, batch_indices, batch_weights):
        if self.params.NEXT_STATE_IN_TRAJECTORY:
            states, actions, rewards, dones, next_states, last_steps = self.unpack_batch(batch)
        else:
            states, actions, rewards, dones, next_states, last_steps = self.unpack_batch_for_n_step(batch, batch_indices)

        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        done_mask = torch.BoolTensor(dones)
        last_steps_v = torch.tensor(last_steps, dtype=torch.float32)
        batch_weights_v = torch.tensor(batch_weights)
        if self.device == torch.device("cuda"):
            states_v = states_v.cuda(non_blocking=True)
            next_states_v = next_states_v.cuda(non_blocking=True)
            actions_v = actions_v.cuda(non_blocking=True)
            rewards_v = rewards_v.cuda(non_blocking=True)
            done_mask = done_mask.cuda(non_blocking=True)
            last_steps_v = last_steps_v.cuda(non_blocking=True)
            batch_weights_v = batch_weights_v.cuda(non_blocking=True)

        actions_v = actions_v.unsqueeze(-1)
        action_values = self.model(states_v).gather(1, actions_v)
        action_values = action_values.squeeze(-1)

        with torch.no_grad():
            next_state_actions = self.model(next_states_v).max(1)[1]
            next_state_actions = next_state_actions.unsqueeze(-1)
            next_state_values = self.target_agent.target_model(next_states_v).gather(1, next_state_actions).squeeze(-1)
            next_state_values[done_mask] = 0.0

            target_action_values = next_state_values.detach() * (self.params.GAMMA ** last_steps_v) + rewards_v

        losses_each = F.smooth_l1_loss(action_values, target_action_values.detach(), reduction='none')
        weighted_losses_v = batch_weights_v.detach() * losses_each

        return weighted_losses_v.mean(), losses_each + 1e-5

    def calc_loss_per_double_dqn_for_omega(self, batch, batch_indices, batch_weights):
        states, actions, rewards, done_mask, next_states = self.unpack_batch_for_omega(batch, batch_indices)

        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        batch_weights_v = torch.tensor(batch_weights)
        if self.device == torch.device("cuda"):
            states_v = states_v.cuda(non_blocking=True)
            next_states_v = next_states_v.cuda(non_blocking=True)
            actions_v = actions_v.cuda(non_blocking=True)
            batch_weights_v = batch_weights_v.cuda(non_blocking=True)

        actions_v = actions_v.unsqueeze(-1)
        action_values = self.model(states_v).gather(1, actions_v)
        action_values = action_values.squeeze(-1)

        with torch.no_grad():
            # for double DQN
            next_state_actions = self.model(next_states_v).max(1)[1]
            next_state_actions = next_state_actions.unsqueeze(-1)
            next_state_values = self.target_agent.target_model(next_states_v).gather(1, next_state_actions).squeeze(-1)

            target_action_values = self.calc_omega_return(
                rewards, done_mask, next_state_values.detach().cpu().numpy()
            )
        target_action_values = torch.tensor(target_action_values, dtype=torch.float32)
        if self.device == torch.device("cuda"):
            target_action_values = target_action_values.cuda(non_blocking=True)

        losses_each = F.smooth_l1_loss(action_values, target_action_values.detach(), reduction='none')
        weighted_losses_v = batch_weights_v.detach() * losses_each

        return weighted_losses_v.mean(), losses_each + 1e-5

    def calc_omega_return(self, rewards, done_mask, next_state_values):
        idx_count = 0
        target_q_values = []
        for batch_idx in range(self.params.BATCH_SIZE):
            n_step_target_list = []
            n_step_reward_sum_list = []
            reward_sum = 0
            gamma = 1
            for idx, reward in enumerate(rewards[batch_idx]):
                reward_sum += gamma * reward
                n_step_reward_sum_list.append(reward_sum)
                gamma *= self.params.GAMMA
            gamma = self.params.GAMMA
            for i in range(len(rewards[batch_idx])):
                n_step_target_list.append(n_step_reward_sum_list[i] + gamma * next_state_values[idx_count] *
                                          (done_mask[batch_idx] if i == len(rewards[batch_idx]) - 1 else 1))
                gamma *= self.params.GAMMA
                idx_count += 1

            avg = sum(n_step_target_list) / len(n_step_target_list)
            max_n_step_target = max(n_step_target_list)
            abs_n_step_target_list = np.abs(n_step_target_list)
            beta = (max(abs_n_step_target_list)-min(abs_n_step_target_list)) / (max(abs_n_step_target_list) + 0.00000001)
            target_q_values.append((1 - beta) * avg + beta * max_n_step_target)

        return target_q_values