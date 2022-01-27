# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
# https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC.py
# PAPER: https://arxiv.org/abs/1812.05905
# https://www.pair.toronto.edu/csc2621-w20/assets/slides/lec4_sac.pdf
# https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/
import math

import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal

from c_models.h_muzero_models import *
from c_models.h_muzero_models import MuzeroModel
from d_agents.agent import Agent
from g_utils.types import AgentMode

# 저장되는 transition이 다르다.
# get action에 num_simulation만큼 반복하는 mcts 구현


class AgentMuZero(Agent):
    def __init__(self, observation_space, action_space, parameter):
        super(AgentMuZero, self).__init__(observation_space, action_space, parameter)

        if isinstance(self.action_space, Discrete):
            self.model = MuzeroModel()
        elif isinstance(self.action_space, Box):
            pass
        else:
            raise ValueError()

    def get_action(self, obs, mode=AgentMode.TRAIN):
        if isinstance(self.action_space, Discrete):
            pass
        elif isinstance(self.action_space, Box):
            pass

    def train_muzero(self, training_steps_v):
        if isinstance(self.action_space, Discrete):
            pass
        elif isinstance(self.action_space, Box):
            pass
        else:
            raise ValueError()


class SelfPlay:
    pass


class MCTS:
    def __init__(self, parameter):
        self.param = parameter

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """

        root = Node(0)
        # 3차원의 영상(atrai)등을 처리하기 위해하는 전처리 작업, observation.shape[0] = 1이 항상 만족한다.
        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(next(model.parameters()).device)
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
            root_predicted_value, self.param.support_size
        ).item()
        reward = support_to_scalar(reward, self.param.support_size).item()
        # TODO : action space의 type에 대한 정리
        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.param.action_space)
        ), "Legal actions should be a subset of the action space."
        # 의문 : child node에 대한 legal actions는 따로 처리 안하는 것인가? 그러면 parent node와 child node 간에 legal action이 다를경우
        #       애시당초 진짜 환경의 policy logits와 일치할 수 없는거 아닌가?
        root.expand(
            legal_actions,  # root node 자신에 대한
            to_play,
            reward,
            policy_logits,
            hidden_state,
        )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.param.root_dirichlet_alpha,
                exploration_fraction=self.param.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            # 해당 state에 action을 했을 경우 얻는 value와 reward
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = support_to_scalar(value, self.config.support_size).item()
            reward = support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                self.config.action_space,
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
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
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
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

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
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac
