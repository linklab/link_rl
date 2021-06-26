# https://github.com/higgsfield/RL-Adventure
import numpy as np
import torch
import torch.nn.functional as F

from codes.a_config._rl_parameters.off_policy.parameter_dqn import DQNActionSelectorType
from codes.c_models.advanced_exploration.curiosity_driven import CuriosityMlpStateEncoder, CuriosityCnnStateEncoder, \
    CuriosityForwardModel, CuriosityInverseModel, intrinsic_curiosity_module_errors
from codes.c_models.discrete_action.dqn_model import DuelingDQNModel
from codes.d_agents.a0_base_agent import float32_preprocessor
from codes.d_agents.off_policy.dqn.dqn_action_selector import EpsilonGreedySomeTimesBlowDQNActionSelector, \
    EpsilonGreedyDQNActionSelector
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.e_utils import rl_utils
from codes.d_agents.actions import ArgmaxActionSelector, EpsilonTracker
from codes.e_utils.names import DeepLearningModelName, AgentMode, EnvironmentName


class AgentDQN(OffPolicyAgent):
    """
    """
    def __init__(self, worker_id, input_shape, action_shape, num_outputs, action_min, action_max, params, device):
        self.__name__ = "AgentDQN"

        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.DUELING_DQN_MLP,
            DeepLearningModelName.DUELING_DQN_CNN,
            DeepLearningModelName.DUELING_DQN_SMALL_CNN
        ]

        super(AgentDQN, self).__init__(
            worker_id=worker_id, params=params, action_shape=action_shape,
            action_min=action_min, action_max=action_max, device=device
        )

        if self.params.DISTRIBUTIONAL:
            self.delta_z = float(self.params.VALUE_MAX - self.params.VALUE_MIN) / (self.params.NUM_SUPPORTS - 1)
            # self.supports.size(): (51,)
            self.supports = torch.linspace(self.params.VALUE_MIN, self.params.VALUE_MAX, self.params.NUM_SUPPORTS).to(device)
        else:
            self.delta_z = None
            self.supports = None

        ################## BEGIN OF ACTION SELECTOR ####################

        if params.TYPE_OF_DQN_ACTION_SELECTOR == DQNActionSelectorType.BASIC_ACTION_SELECTOR:
            self.train_action_selector = EpsilonGreedyDQNActionSelector(epsilon=params.EPSILON_INIT)
        elif params.TYPE_OF_DQN_ACTION_SELECTOR == DQNActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR:
            self.train_action_selector = EpsilonGreedySomeTimesBlowDQNActionSelector(
                epsilon=params.EPSILON_INIT, min_blowing_action_idx=0, max_blowing_action_idx=num_outputs - 1,
                params=params
            )
        elif params.TYPE_OF_DQN_ACTION_SELECTOR == DQNActionSelectorType.NOISY_NET_ACTION_SELECTOR:
            if self.params.DISTRIBUTIONAL:
                self.train_action_selector = ArgmaxActionSelector(supports_numpy=self.supports.cpu().numpy())
            else:
                self.train_action_selector = ArgmaxActionSelector()
        else:
            raise ValueError()

        if self.params.DISTRIBUTIONAL:
            self.test_and_play_action_selector = ArgmaxActionSelector(supports_numpy=self.supports.cpu().numpy())
        else:
            self.test_and_play_action_selector = ArgmaxActionSelector()

        if params.ENVIRONMENT_ID in [EnvironmentName.TRADE_V0]:
            # main 에서 action_selector 할당
            self.train_action_selector = None
            self.test_and_play_action_selector = None

         ################## END OF ACTION SELECTOR ####################

        if params.TYPE_OF_DQN_ACTION_SELECTOR == DQNActionSelectorType.NOISY_NET_ACTION_SELECTOR:
            self.epsilon_tracker = None
        else:
            self.epsilon_tracker = EpsilonTracker(
                action_selector=self.train_action_selector,
                eps_start=params.EPSILON_INIT,
                eps_final=params.EPSILON_MIN,
                eps_last_frames=params.EPSILON_MIN_STEP
            )

        self.model = DuelingDQNModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.target_model = DuelingDQNModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.test_model = DuelingDQNModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        if params.CURIOSITY_DRIVEN:
            if params.DEEP_LEARNING_MODEL == DeepLearningModelName.DUELING_DQN_MLP:
                num_inputs = input_shape[0]
                self.curiosity_state_encoder = CuriosityMlpStateEncoder(
                    num_inputs=num_inputs, params=params, device=device
                )
            elif params.DEEP_LEARNING_MODEL in [DeepLearningModelName.DUELING_DQN_CNN, DeepLearningModelName.DUELING_DQN_SMALL_CNN]:
                self.curiosity_state_encoder = CuriosityCnnStateEncoder(
                    input_shape=input_shape, params=params, device=device
                )
            else:
                raise ValueError()

            self.curiosity_forward_model = CuriosityForwardModel(
                self.curiosity_state_encoder.encoded_state_size, num_outputs, device
            )
            self.curiosity_inverse_model = CuriosityInverseModel(
                self.curiosity_state_encoder.encoded_state_size, num_outputs, device
            )

            parameters = list(self.model.base.parameters())
            parameters += list(self.curiosity_state_encoder.parameters())
            parameters += list(self.curiosity_forward_model.parameters())
            parameters += list(self.curiosity_inverse_model.parameters())
        else:
            self.curiosity_state_encoder = None
            self.curiosity_forward_model = None
            self.curiosity_inverse_model = None
            parameters = self.model.base.parameters()

        self.optimizer = rl_utils.get_optimizer(
            parameters=parameters,
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

        if self.agent_mode == AgentMode.TRAIN:
            q_v = self.model(states)
            q = q_v.detach().cpu().numpy()
            actions = self.train_action_selector(q)
        else:
            q_v = self.test_model(states)
            q = q_v.detach().cpu().numpy()
            actions = self.test_and_play_action_selector(q)

        return actions, agent_states

    def on_train(self, step_idx):
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
                if self.params.DISTRIBUTIONAL:
                    loss_v = self.calc_loss_distributional_dqn(batch)
                else:
                    if self.params.DOUBLE:
                        loss_v = self.calc_loss_double_dqn(batch)
                    else:
                        loss_v = self.calc_loss_dqn(batch)

        loss_v.backward()

        self.optimizer.step()

        if step_idx % self.params.TARGET_NET_SYNC_STEP_PERIOD < self.params.TRAIN_STEP_FREQ:
            self.target_model.sync(self.model)

        gradients = self.model.get_gradients_for_current_parameters()

        #self.model.check_gradient_nan_or_zero(gradients)

        if self.params.TYPE_OF_DQN_ACTION_SELECTOR == DQNActionSelectorType.NOISY_NET_ACTION_SELECTOR:
            self.model.base.reset_noise()  # Pick a new noise vector (until next optimisation step)
            self.target_model.base.reset_noise()

        return gradients, loss_v.detach().item(), None

    def unpack_batch(self, batch):
        states, actions, rewards, dones, last_states, last_steps = [], [], [], [], [], []

        for exp in batch:
            state = np.array(exp.state, copy=False)
            states.append(state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(state)  # the result will be masked anyway
            else:
                last_states.append(np.array(exp.last_state, copy=False))
            last_steps.append(exp.last_step)
        return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(last_states, copy=False), np.array(last_steps)

    # def unpack_batch_for_n_step(self, batch, batch_indices):
    #     states, actions, rewards, dones, next_states, last_steps = [], [], [], [], [], []
    #     for idx, exp in enumerate(batch):
    #         state = np.array(exp.state, copy=False)
    #         states.append(state)
    #         actions.append(exp.action)
    #
    #         n_step_rewards = 0
    #         gamma = 1
    #         current_exp = exp
    #         for i in range(self.params.N_STEP):
    #             n_step_rewards += gamma * current_exp.reward
    #             next_exp = self.buffer.buffer[(batch_indices[idx] + i + 1) % self.params.REPLAY_BUFFER_SIZE]
    #
    #             if current_exp.done:
    #                 rewards.append(n_step_rewards)
    #                 next_states.append(np.array(next_exp.state, copy=False))
    #                 dones.append(True)
    #                 last_steps.append(i + 1)
    #                 break
    #             else:
    #                 if i == self.params.N_STEP - 1:
    #                     rewards.append(n_step_rewards)
    #                     next_states.append(np.array(next_exp.state, copy=False))
    #                     dones.append(False)
    #                     last_steps.append(i + 1)
    #
    #             current_exp = next_exp
    #             gamma *= self.params.GAMMA
    #
    #     return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
    #            np.array(dones, dtype=np.uint8), np.array(next_states, copy=False), np.array(last_steps)

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

        if self.params.CURIOSITY_DRIVEN:
            intrinsic_reward_v, intrinsic_loss_v = self.get_intrinsic_curiosity_module_info(
                states_v, actions_v, next_states_v
            )
            rewards_v += intrinsic_reward_v
        else:
            intrinsic_reward_v, intrinsic_loss_v = None, None

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
            next_state_action_values = self.target_model(next_states_v).max(1)[0]
            next_state_action_values[done_mask] = 0.0

        target_action_values = rewards_v + next_state_action_values.detach() * (self.params.GAMMA ** last_steps_v)

        loss_v = F.smooth_l1_loss(action_values, target_action_values)

        if self.params.CURIOSITY_DRIVEN:
            loss_v = intrinsic_loss_v + self.params.CURIOSITY_DRIVEN_LAMBDA * loss_v

        return loss_v

    def calc_loss_distributional_dqn(self, batch):
        states, actions, rewards, dones, next_states, last_steps = self.unpack_batch(batch)

        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        done_masks_v = torch.BoolTensor(dones)

        if self.device == torch.device("cuda"):
            states_v = states_v.cuda(non_blocking=True)
            next_states_v = next_states_v.cuda(non_blocking=True)
            actions_v = actions_v.cuda(non_blocking=True)
            rewards_v = rewards_v.cuda(non_blocking=True)
            done_masks_v = done_masks_v.cuda(non_blocking=True)

        if self.params.CURIOSITY_DRIVEN:
            intrinsic_reward_v, intrinsic_loss_v = self.get_intrinsic_curiosity_module_info(
                states_v, actions_v, next_states_v
            )
            rewards_v += intrinsic_reward_v
        else:
            intrinsic_reward_v, intrinsic_loss_v = None, None

        # proj_dist.size(): (32, 51)
        proj_dist = self.projection_distribution_for_next_state(next_states_v, rewards_v, done_masks_v)

        # dist.size(): (32, 2, 51)
        dist = self.model(states_v)

        # actions_v.unsqueeze(1).unsqueeze(1).size(): (32, 1, 1)
        # actions: (32, 1, 51)
        actions = actions_v.unsqueeze(1).unsqueeze(1).expand(self.params.BATCH_SIZE, 1, self.params.NUM_SUPPORTS)

        # dist.size(): (32, 51)
        dist = dist.gather(1, actions).squeeze(1)

        dist.data.clamp_(0.01, 0.99)
        # print((proj_dist * dist.log()).sum(1), "!!!!!!!!!!!!1")
        loss_v = -1.0 * (proj_dist * dist.log()).sum(1).mean()

        if self.params.CURIOSITY_DRIVEN:
            loss_v = intrinsic_loss_v + self.params.CURIOSITY_DRIVEN_LAMBDA * loss_v

        return loss_v

    def calc_loss_double_dqn(self, batch):
        states, actions, rewards, dones, next_states, last_steps = self.unpack_batch(batch)

        states_v = torch.tensor(states)
        next_states_v = torch.tensor(next_states)
        actions_v = torch.tensor(actions)
        rewards_v = torch.tensor(rewards)
        done_masks_v = torch.BoolTensor(dones)
        last_steps_v = torch.tensor(last_steps)

        # print(states_v.size(), next_states_v.size(), actions_v.size(), rewards_v.size(), done_masks_v.size(), last_steps_v.size(), "!!!!")

        if self.device == torch.device("cuda"):
            states_v = states_v.cuda(non_blocking=True)
            next_states_v = next_states_v.cuda(non_blocking=True)
            actions_v = actions_v.cuda(non_blocking=True)
            rewards_v = rewards_v.cuda(non_blocking=True)
            done_masks_v = done_masks_v.cuda(non_blocking=True)
            last_steps_v = last_steps_v.cuda(non_blocking=True)

        if self.params.CURIOSITY_DRIVEN:
            intrinsic_reward_v, intrinsic_loss_v = self.get_intrinsic_curiosity_module_info(
                states_v, actions_v, next_states_v
            )
            rewards_v += intrinsic_reward_v
        else:
            intrinsic_reward_v, intrinsic_loss_v = None, None

        actions_v = actions_v.unsqueeze(-1)
        action_values = self.model(states_v).gather(1, actions_v)
        action_values = action_values.squeeze(-1)

        with torch.no_grad():
            next_state_acts = self.model(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_vals = self.target_model(next_states_v).gather(1, next_state_acts).squeeze(-1)
            next_state_vals[done_masks_v] = 0.0

        exp_sa_vals = next_state_vals.detach() * (self.params.GAMMA ** last_steps_v) + rewards_v
        loss_v = F.smooth_l1_loss(action_values, exp_sa_vals)

        if self.params.CURIOSITY_DRIVEN:
            loss_v = intrinsic_loss_v + self.params.CURIOSITY_DRIVEN_LAMBDA * loss_v

        return loss_v

    def calc_loss_per_double_dqn(self, batch, batch_indices, batch_weights):
        # if self.params.NEXT_STATE_IN_TRAJECTORY:
        #     states, actions, rewards, dones, next_states, last_steps = self.unpack_batch(batch)
        # else:
        #     states, actions, rewards, dones, next_states, last_steps = self.unpack_batch_for_n_step(batch, batch_indices)

        states, actions, rewards, dones, next_states, last_steps = self.unpack_batch(batch)

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

        if self.params.CURIOSITY_DRIVEN:
            intrinsic_reward_v, intrinsic_loss_v = self.get_intrinsic_curiosity_module_info(
                states_v, actions_v, next_states_v
            )
            rewards_v += intrinsic_reward_v
        else:
            intrinsic_reward_v, intrinsic_loss_v = None, None

        actions_v = actions_v.unsqueeze(-1)
        action_values = self.model(states_v).gather(1, actions_v)
        action_values = action_values.squeeze(-1)

        with torch.no_grad():
            next_state_actions = self.model(next_states_v).max(1)[1]
            next_state_actions = next_state_actions.unsqueeze(-1)
            next_state_action_values = self.target_model(next_states_v).gather(1, next_state_actions).squeeze(-1)
            next_state_action_values[done_mask] = 0.0

            target_action_values = next_state_action_values.detach() * (self.params.GAMMA ** last_steps_v) + rewards_v

        losses_each = F.smooth_l1_loss(action_values, target_action_values.detach(), reduction='none')
        loss_v = batch_weights_v.detach() * losses_each

        if self.params.CURIOSITY_DRIVEN:
            loss_v = intrinsic_loss_v + self.params.CURIOSITY_DRIVEN_LAMBDA * loss_v

        return loss_v.mean(), losses_each + 1e-5

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

        if self.params.CURIOSITY_DRIVEN:
            raise ValueError()

        actions_v = actions_v.unsqueeze(-1)
        action_values = self.model(states_v).gather(1, actions_v)
        action_values = action_values.squeeze(-1)

        with torch.no_grad():
            # for double DQN
            next_state_actions = self.model(next_states_v).max(1)[1]
            next_state_actions = next_state_actions.unsqueeze(-1)
            next_state_action_values = self.target_model(next_states_v).gather(1, next_state_actions).squeeze(-1)

            target_action_values = self.calc_omega_return(
                rewards, done_mask, next_state_action_values.detach().cpu().numpy()
            )
        target_action_values = torch.tensor(target_action_values, dtype=torch.float32)
        if self.device == torch.device("cuda"):
            target_action_values = target_action_values.cuda(non_blocking=True)

        losses_each = F.smooth_l1_loss(action_values, target_action_values.detach(), reduction='none')
        weighted_losses_v = batch_weights_v.detach() * losses_each

        return weighted_losses_v.mean(), losses_each + 1e-5

    def calc_omega_return(self, rewards, done_mask, next_state_action_values):
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
                n_step_target_list.append(n_step_reward_sum_list[i] + gamma * next_state_action_values[idx_count] *
                                          (done_mask[batch_idx] if i == len(rewards[batch_idx]) - 1 else 1))
                gamma *= self.params.GAMMA
                idx_count += 1

            avg = sum(n_step_target_list) / len(n_step_target_list)
            max_n_step_target = max(n_step_target_list)
            abs_n_step_target_list = np.abs(n_step_target_list)
            beta = (max(abs_n_step_target_list)-min(abs_n_step_target_list)) / (max(abs_n_step_target_list) + 0.00000001)
            target_q_values.append((1 - beta) * avg + beta * max_n_step_target)

        return target_q_values

    def projection_distribution_for_next_state(self, next_states, rewards, done_masks):
        batch_size = next_states.size(0)

        # self.target_model(next_states).size(): [32, 2, 51]
        # next_dist.size(): [32, 2, 51]
        next_dist = self.target_model(next_states) * self.supports
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))

        # next_dist.size(): [32, 51]
        next_dist = next_dist.gather(1, next_action).squeeze(1)
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        done_masks = torch.tensor(done_masks, dtype=torch.uint8).unsqueeze(1).expand_as(next_dist)
        Tz = rewards + (1 - done_masks) * 0.99 * self.supports

        Tz = Tz.clamp(min=self.params.VALUE_MIN, max=self.params.VALUE_MAX)
        b = (Tz - self.params.VALUE_MIN) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(
            0,
            (batch_size - 1) * self.params.NUM_SUPPORTS,
            batch_size
        ).long().unsqueeze(1).expand(batch_size, self.params.NUM_SUPPORTS).to(self.device)

        proj_dist = torch.zeros(next_dist.size()).to(self.device)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def get_intrinsic_curiosity_module_info(self, state_batch, action_batch, next_state_batch):
        # forward_pred_loss.shape: [32, 1]
        # inverse_pred_loss.shape: [32, 1]
        forward_pred_loss, inverse_pred_loss = intrinsic_curiosity_module_errors(
            self.curiosity_state_encoder, self.curiosity_forward_model, self.curiosity_inverse_model,
            state_batch, action_batch, next_state_batch,
        )

        intrinsic_reward_v = (1. / self.params.CURIOSITY_DRIVEN_ETA) * forward_pred_loss.squeeze(dim=-1)

        intrinsic_loss_v = (1 - self.params.CURIOSITY_DRIVEN_BETA) * inverse_pred_loss + \
                           self.params.CURIOSITY_DRIVEN_BETA * forward_pred_loss
        intrinsic_loss_v = intrinsic_loss_v.sum() / intrinsic_loss_v.flatten().shape[0]

        # intrinsic_reward_v.shape: [32]
        # intrinsic_loss_v.shape: []
        return intrinsic_reward_v, intrinsic_loss_v
