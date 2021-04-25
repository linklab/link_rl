import random
import numpy as np

from codes.d_agents.actions import ActionSelector, ArgmaxActionSelector


class EpsilonGreedyDQNActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, default_action_selector=None):
        self.epsilon = epsilon
        self.default_action_selector = default_action_selector if default_action_selector is not None else ArgmaxActionSelector()

    def __call__(self, q_values):
        assert isinstance(q_values, np.ndarray)
        batch_size, n_actions = q_values.shape
        actions = self.default_action_selector(q_values)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions


class EpsilonGreedySomeTimesBlowDQNActionSelector(ActionSelector):
    #TODO: max_blowing_action_idx
    def __init__(
            self, epsilon=0.05, blowing_action_rate=0.0002,
            min_blowing_action_idx=0, max_blowing_action_idx=-1, default_action_selector=None
    ):
        self.epsilon = epsilon
        self.default_action_selector = default_action_selector if default_action_selector is not None else ArgmaxActionSelector()

        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action_idx = min_blowing_action_idx
        self.max_blowing_action_idx = max_blowing_action_idx
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        if self.time_steps == 0:
            print("next_time_steps_of_random_blowing_action: {0}".format(
                self.next_time_steps_of_random_blowing_action
            ))

        self.time_steps += 1
        batch_size, n_actions = scores.shape
        actions = self.default_action_selector(scores)

        if self.time_steps >= self.next_time_steps_of_random_blowing_action:
            actions = np.random.choice(
                a=[self.min_blowing_action_idx, self.max_blowing_action_idx], size=actions.shape
            )

            # actions += np.random.uniform(
            #     low=self.min_blowing_action, high=self.max_blowing_action, size=actions.shape
            # )

            self.next_time_steps_of_random_blowing_action = self.time_steps + int(random.expovariate(self.blowing_action_rate))
            print("Internal Blowing Action: {0}, next_time_steps_of_random_blowing_action: {1}".format(
                actions,
                self.next_time_steps_of_random_blowing_action
            ))
        else:
            mask = np.random.random(size=batch_size) < self.epsilon
            rand_actions = np.random.choice(n_actions, sum(mask))
            actions[mask] = rand_actions
        return actions