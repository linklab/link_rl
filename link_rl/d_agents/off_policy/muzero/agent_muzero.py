# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
# https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC.py
# PAPER: https://arxiv.org/abs/1812.05905
# https://www.pair.toronto.edu/csc2621-w20/assets/slides/lec4_sac.pdf
# https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/
import math

import torch.optim as optim
import numpy as np
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box

from link_rl.c_models.h_muzero_models import *
from link_rl.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from link_rl.g_utils.types import AgentMode

# 저장되는 transition이 다르다.
# get action에 num_simulation만큼 반복하는 mcts 구현


class AgentMuZero(OffPolicyAgent):
    def __init__(self, observation_space, action_space, config):
        super(AgentMuZero, self).__init__(observation_space, action_space, config)

        if isinstance(self.action_space, Discrete):
            self.model = DiscreteMuzeroModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, config=config
            )
        elif isinstance(self.action_space, Box):
            raise ValueError()
        else:
            raise ValueError()

        self.config = config
        self.to_plays = []
        self.legal_actions = []
        self.roots = None
        self.mcts_extra_infos = None
        self.temperature = mp.Value('d', 1.0)
        self.value_loss = mp.Value('d', 1.0)
        self.policy_loss = mp.Value('d', 1.0)
        self.reward_loss = mp.Value('d', 1.0)
        self.loss = mp.Value('d', 1.0)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

    def get_action(self, observations, mode=AgentMode.TRAIN):
        action = []
        with torch.no_grad():
            if isinstance(self.action_space, Discrete):
                self.roots, self.mcts_extra_infos = MCTS(self.config).run(
                    model=self.model,
                    observations=observations,
                    legal_actions=self.legal_actions,
                    to_plays=self.to_plays,
                    action_space=self.action_space,
                    add_exploration_noise=True,
                )
                # child_visit_counts's shape = (n_vectorized, n_root's_child)
                child_visit_counts = np.array(
                   [self.mcts_extra_infos[env_id]["child_visit_counts"] for env_id in range(len(self.roots))], dtype="int32"
                )
                # actions's shape = (n_vectorized, n_root's_child)
                actions = np.array(
                   [self.mcts_extra_infos[env_id]["actions"] for env_id in range(len(self.roots))]
                )
                # TODO : numpy batch 연산
                if mode == AgentMode.TRAIN:
                    visit_count_distributions = child_visit_counts ** (1 / self.temperature.value)
                    visit_count_distributions = visit_count_distributions / \
                                                np.sum(visit_count_distributions, axis=-1)[:, np.newaxis]
                    for i in range(self.config.N_VECTORIZED_ENVS):
                        action.append(np.random.choice(actions[i], p=visit_count_distributions[i]))
                elif mode == AgentMode.TEST:
                    for i in range(self.config.N_VECTORIZED_ENVS):
                        action.append(actions[i][np.argmax(child_visit_counts[i])])

            elif isinstance(self.action_space, Box):
                pass

        return np.asarray(action)

    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])

        for episode_idx, episode_history in zip(self.episode_idxs, self.episode_historys):
            state_index = np.random.choice(len(episode_history.root_values_history))

            values, rewards, policies, actions = self.make_target(
                episode_history, state_index
            )

            index_batch.append([episode_idx, state_index])
            observation_batch.append(episode_history.observation_history[state_index])
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.NUM_UNROLL_STEPS,
                        len(episode_history.action_history) - state_index,
                    )
                ]
                * len(actions)
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                gradient_scale_batch,
            ),
        )

    def compute_target_value(self, episode_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.N_STEP
        if bootstrap_index <= len(episode_history.root_values_history)-1:
            root_values = (episode_history.root_values_history)
            last_step_value = (
                root_values[bootstrap_index]
                if episode_history.to_play_history[bootstrap_index]
                   == episode_history.to_play_history[index]
                else -root_values[bootstrap_index]
            )
            value = last_step_value * self.config.GAMMA ** self.config.N_STEP
        else:
            value = 0

        for i, reward in enumerate(
                episode_history.reward_history[index: bootstrap_index]
        ):
            # The value is oriented from the perspective of the current player
            value += (
                         reward
                         if episode_history.to_play_history[index]
                            == episode_history.to_play_history[index + i]
                         else -reward
                     ) * self.config.GAMMA ** i

        return value

    def make_target(self, episode_history, state_index):
        """
        make only one episode's target not batch
        """
        target_values, target_rewards, target_policies, actions = [], [], [], []

        for current_index in range(
                state_index, state_index + self.config.NUM_UNROLL_STEPS + 1
        ):
            value = self.compute_target_value(episode_history, current_index)

            if current_index <= len(episode_history.root_values_history)-1:
                target_values.append(value)
                target_rewards.append(episode_history.reward_history[current_index])
                target_policies.append(episode_history.child_visits_history[current_index])
                actions.append(episode_history.action_history[current_index])
            # elif current_index == len(episode_history.root_values_history):
            #     target_values.append(0)
            #     target_rewards.append(episode_history.reward_history[current_index-1])
            #     # Uniform policy
            #     target_policies.append(
            #         [
            #             1 / len(episode_history.child_visits_history[0])
            #             for _ in range(len(episode_history.child_visits_history[0]))
            #         ]
            #     )
            #     actions.append(episode_history.action_history[current_index-1])
            else:
                # States past the end of games are treated as absorbing states
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(episode_history.child_visits_history[0])
                        for _ in range(len(episode_history.child_visits_history[0]))
                    ]
                )
                actions.append(np.random.choice(list(range(self.action_space.n))))

        return target_values, target_rewards, target_policies, actions

    def get_stacked_observations(self, index, num_stacked_observations, observation_history, action_history):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(observation_history)

        stacked_observations = observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = np.concatenate(
                    (
                        observation_history[past_observation_index],
                        [
                            np.ones_like(stacked_observations[0])
                            * action_history[past_observation_index + 1]
                        ],
                    )
                )
            else:
                previous_observation = np.concatenate(
                    (
                        np.zeros_like(observation_history[index]),
                        [np.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = np.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations

    def store_search_statistics(self, root, action_space, child_visits, root_values):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            root_values.append(root.value())
        else:
            root_values.append(None)

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.config.MAX_TRAINING_STEPS:
            self.temperature.value = 1.0
        elif trained_steps < 0.75 * self.config.MAX_TRAINING_STEPS:
            self.temperature.value = 0.5
        else:
            self.temperature.value = 0.25

    def loss_function(
        self,
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        # torch.Size([128, 21]) torch.Size([128, 21]) torch.Size([128, 21]) torch.Size([128, 21])
        # torch.Size([128, 2]) torch.Size([128, 2]) !!!!!
        # print(
        #     target_value.shape, value.shape, target_reward.shape, reward.shape,
        #     target_policy.shape, policy_logits.shape, "!!!!!"
        # )
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=1)(value)).sum(1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=1)(reward)).sum(1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(1)

        # torch.Size([128]) torch.Size([128]) torch.Size([128]) !@!@
        # print(value_loss.shape, reward_loss.shape, policy_loss.shape, "!@!@")

        return value_loss, reward_loss, policy_loss

    def train_muzero(self, training_steps_v):
        count_training_steps = 0
        for _ in range(20):
            batch = self.get_batch()
            if isinstance(self.action_space, Discrete):
                (
                    observation_batch,
                    action_batch,
                    target_value,
                    target_reward,
                    target_policy,
                    gradient_scale_batch,
                ) = batch[-1]

                # Keep values as scalars for calculating the priorities for the prioritized replay
                target_value_scalar = np.array(target_value, dtype="float32")
                priorities = np.zeros_like(target_value_scalar)

                observation_batch = torch.tensor(observation_batch).float().to(self.config.DEVICE)
                action_batch = torch.tensor(action_batch).long().to(self.config.DEVICE).unsqueeze(-1)
                target_value = torch.tensor(target_value).float().to(self.config.DEVICE)
                target_reward = torch.tensor(target_reward).float().to(self.config.DEVICE)
                target_policy = torch.tensor(target_policy).float().to(self.config.DEVICE)
                gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(self.config.DEVICE)

                # torch.Size([128, 4]) torch.Size([128, 6, 1]) torch.Size([128, 6]) torch.Size([128, 6])
                # torch.Size([128, 6, 2]) torch.Size([128, 6]) !!!!!!!!!!
                # print(
                #     observation_batch.shape, action_batch.shape, target_value.shape, target_reward.shape,
                #     target_policy.shape, gradient_scale_batch.shape, "!!!!!!!!!!"
                # )

                # observation_batch: batch, channels, height, width
                # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
                # target_value: batch, num_unroll_steps+1
                # target_reward: batch, num_unroll_steps+1
                # target_policy: batch, num_unroll_steps+1, len(action_space)
                # gradient_scale_batch: batch, num_unroll_steps+1

                target_value = scalar_to_support(target_value, self.config.SUPPORT_SIZE)
                target_reward = scalar_to_support(target_reward, self.config.SUPPORT_SIZE)
                # target_value: batch, num_unroll_steps+1, 2*support_size+1
                # target_reward: batch, num_unroll_steps+1, 2*support_size+1

                ## Generate predictions
                value, reward, policy_logits, hidden_state = self.model.initial_inference(
                    observation_batch
                )

                # torch.Size([128, 21]) torch.Size([128, 21]) torch.Size([128, 2]) torch.Size([128, 128]) ##$$#$#$#$#$
                # print(value.shape, reward.shape, policy_logits.shape, hidden_state.shape, "##$$#$#$#$#$")

                predictions = [(value, reward, policy_logits)]
                for i in range(1, action_batch.shape[1]):
                    value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                        hidden_state, action_batch[:, i]
                    )
                    # TODO : 이해하
                    # Scale the gradient at the start of the dynamics function (See paper appendix Training)
                    hidden_state.register_hook(lambda grad: grad * 0.5)
                    predictions.append((value, reward, policy_logits))
                # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

                ## Compute losses
                value_loss, reward_loss, policy_loss = (0, 0, 0)
                value, reward, policy_logits = predictions[0]
                # Ignore reward loss for the first batch step

                current_value_loss, _, current_policy_loss = self.loss_function(
                    value.squeeze(-1),
                    reward.squeeze(-1),
                    policy_logits,
                    target_value[:, 0],  # current observation에 대한 value distribution
                    target_reward[:, 0],  # current observation에 대한 reward distribution
                    target_policy[:, 0]  # current observation에 대한 policy distribution
                )

                value_loss += current_value_loss
                policy_loss += current_policy_loss

                # print(value.squeeze(-1).shape, reward.squeeze(-1).shape, policy_logits.shape, target_value[:, 0].shape,
                #       target_reward[:, 0].shape, target_policy[:, 0].shape, current_value_loss.shape,
                #       current_policy_loss.shape, value_loss.shape, policy_loss.shape, "!!!!!!!!!")

                for i in range(1, len(predictions)):
                    value, reward, policy_logits = predictions[i]
                    (
                        current_value_loss,
                        current_reward_loss,
                        current_policy_loss,
                    ) = self.loss_function(
                        value.squeeze(-1),
                        reward.squeeze(-1),
                        policy_logits,
                        target_value[:, i],
                        target_reward[:, i],
                        target_policy[:, i],
                    )

                    # # Scale gradient by the number of unroll steps (See paper appendix Training)
                    current_value_loss.register_hook(
                        lambda grad: grad / gradient_scale_batch[:, i]
                    )
                    current_reward_loss.register_hook(
                        lambda grad: grad / gradient_scale_batch[:, i]
                    )
                    current_policy_loss.register_hook(
                        lambda grad: grad / gradient_scale_batch[:, i]
                    )

                    value_loss += current_value_loss
                    reward_loss += current_reward_loss
                    policy_loss += current_policy_loss

                # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
                loss_each = value_loss * self.config.VALUE_LOSS_WEIGHT + reward_loss + policy_loss
                # Mean over batch dimension (pseudocode do a sum)
                loss = loss_each.mean()

                self.value_loss.value = value_loss.mean()
                self.reward_loss.value = reward_loss.mean()
                self.policy_loss.value = policy_loss.mean()
                self.loss.value = loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            elif isinstance(self.action_space, Box):
                pass
            else:
                raise ValueError()
            count_training_steps += 1
        return count_training_steps, loss_each

# TODO : batch MCTS 구현하기
class MCTS:
    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observations,
        legal_actions,
        to_plays,
        add_exploration_noise,  # type(add_exploration_noise) = bool
        action_space,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        action_space = list(range(action_space.n))
        roots = [None for _ in range(self.config.N_VECTORIZED_ENVS)]
        extra_infos = [None for _ in range(self.config.N_VECTORIZED_ENVS)]
        for env_id, (observation, legal_action, to_play) in enumerate(zip(observations, legal_actions, to_plays)):
            root = Node(0)
            # 3차원의 영상(atrai)등을 처리하기 위해하는 전처리 작업, observation.shape[0] = 1이 항상 만족한다.
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(self.config.DEVICE)
            )
            # support_to_scalar ==> muzero에서는 reward나 value도 distribution형태에서 추출하기 때문에
            #                       distribution에서 scalar하나를 뽑는 과정
            (
                root_predicted_value,  # cartegorical distribution, root_node 자기 자신에 대한 value
                reward,                # cartegorical distribution, root_node 자기 자신에 대한 reward
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = support_to_scalar(
                root_predicted_value, self.config.SUPPORT_SIZE
            ).item()
            reward = support_to_scalar(reward, self.config.SUPPORT_SIZE).item()
            # TODO : action space의 type에 대한 정리
            assert (
                legal_action
            ), f"Legal actions should not be an empty array. Got {legal_action}."
            assert set(legal_action).issubset(
                set(action_space)
            ), "Legal actions should be a subset of the action space."
            # 의문 : child node에 대한 legal actions는 따로 처리 안하는 것인가? 그러면 parent node와 child node 간에 legal action이 다를경우
            #       애시당초 진짜 환경의 policy logits와 일치할 수 없는거 아닌가?
            root.expand(
                legal_action,  # root node 자신에 대한 reward(0), hidden_state
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            if add_exploration_noise:
                root.add_exploration_noise(
                    dirichlet_alpha=self.config.ROOT_DIRCHLET_ALPHA,
                    exploration_fraction=self.config.ROOT_EXPLORATION_FRACTION,
                )

            min_max_stats = MinMaxStats()

            max_tree_depth = 0
            for _ in range(self.config.NUM_SIMULATION):
                virtual_to_play = to_play[0]
                node = root
                search_path = [node]
                current_tree_depth = 0

                while node.expanded():
                    current_tree_depth += 1
                    action, node = self.select_child(node, min_max_stats)
                    search_path.append(node)

                    # Players play turn by turn
                    if virtual_to_play + 1 < len(self.config.PLAYERS):
                        virtual_to_play = self.config.PLAYERS[virtual_to_play + 1]
                    else:
                        virtual_to_play = self.config.PLAYERS[0]

                # Inside the search tree we use the dynamics function to obtain the next hidden
                # state given an action and the previous hidden state
                parent = search_path[-2]  # child = search_path[-1]
                # 해당 state에 action을 했을 경우 얻는 value와 reward
                value, reward, policy_logits, hidden_state = model.recurrent_inference(
                    parent.hidden_state,
                    torch.tensor([[action]]).to(parent.hidden_state.device),
                )
                value = support_to_scalar(value, self.config.SUPPORT_SIZE).item()
                reward = support_to_scalar(reward, self.config.SUPPORT_SIZE).item()
                node.expand(
                    action_space,
                    virtual_to_play,
                    reward,
                    policy_logits,
                    hidden_state,
                )

                self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

                max_tree_depth = max(max_tree_depth, current_tree_depth)

            extra_info = {
                "max_tree_depth": max_tree_depth,
                "root_predicted_value": root_predicted_value,
                "child_visit_counts": [child.visit_count for child in root.children.values()],
                "actions": [action for action in root.children.keys()]
            }
            roots[env_id] = root
            extra_infos[env_id] = extra_info
        return roots, extra_infos

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()  # node.children = {action, 이 action으로 선택되는 child_node}
        )
        action = np.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.PB_C_BASE + 1) / self.config.PB_C_BASE
            )
            + self.config.PB_C_INIT
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.GAMMA
                * (child.value() if len(self.config.PLAYERS) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.PLAYERS) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.GAMMA * node.value())

                value = node.reward + self.config.GAMMA * value

        elif len(self.config.PLAYERS) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.GAMMA * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.GAMMA * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        # 갸능한 모든 action에 대한 chile node 생성
        for action, p in policy.items():
            # 왜 child node를 생성할 때 prior 자리에 그 행동을 뽑을 확률이 들어가나.
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

