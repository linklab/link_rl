import random
import numpy as np

from codes.d_agents.actions import ActionSelector, ArgmaxActionSelector


class EpsilonGreedyDQNActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon
        self.default_action_selector = ArgmaxActionSelector()

    def select_action(self, q_values):
        batch_size, n_actions = q_values.shape
        actions = self.default_action_selector(q_values)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(a=n_actions, size=sum(mask))
        actions[mask] = rand_actions
        return actions

    def __call__(self, q_values):
        assert isinstance(q_values, np.ndarray)
        return self.select_action(q_values)


class EpsilonGreedySomeTimesBlowDQNActionSelector(EpsilonGreedyDQNActionSelector):
    def __init__(
            self, epsilon=0.05, blowing_action_rate=0.0002,
            min_blowing_action_idx=None, max_blowing_action_idx=None, params=None
    ):
        super(EpsilonGreedySomeTimesBlowDQNActionSelector, self).__init__(epsilon=epsilon)

        self.blowing_action_rate = blowing_action_rate
        self.min_blowing_action_idx = min_blowing_action_idx
        self.max_blowing_action_idx = max_blowing_action_idx
        self.time_steps = 0
        self.next_time_steps_of_random_blowing_action = int(random.expovariate(self.blowing_action_rate))
        self.params = params

    def __call__(self, q_values):
        assert isinstance(q_values, np.ndarray)
        if self.time_steps == 0:
            print("next_time_steps_of_random_blowing_action: {0}".format(
                self.next_time_steps_of_random_blowing_action
            ))

        self.time_steps += 1

        if self.time_steps >= self.next_time_steps_of_random_blowing_action:
            actions = self.default_action_selector(q_values)
            actions = np.random.choice(
                a=[self.min_blowing_action_idx, self.max_blowing_action_idx], size=actions.shape
            )

            # actions += np.random.uniform(
            #     low=self.min_blowing_action, high=self.max_blowing_action, size=actions.shape
            # )

            self.next_time_steps_of_random_blowing_action = self.time_steps + int(random.expovariate(self.blowing_action_rate))
            print("[{0:6}/{1}] Internal Blowing Action: {2}, next_time_steps_of_random_blowing_action: {3}".format(
                self.time_steps,
                self.params.MAX_GLOBAL_STEP,
                actions,
                self.next_time_steps_of_random_blowing_action
            ))
        else:
            actions = self.select_action(q_values=q_values)
        return actions