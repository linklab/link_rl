from or_gym.envs.classic_or import KnapsackEnv, BoundedKnapsackEnv
import numpy as np


class CustomBoundedKnapsackEnv(BoundedKnapsackEnv):
    def __init__(self, *args, **kwargs):
        super(CustomBoundedKnapsackEnv, self).__init__(*args, **kwargs)
        self.previous_action_mask = None

    def reset(self):
        state = super().reset()

        print(state)

        if self.mask:
            updated_state = state['state']
            self.previous_action_mask = state['action_mask']
        else:
            updated_state = []
            for l in state:
                updated_state.extend(l)

        return updated_state

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        print(action, next_state, reward, done)

        if self.mask:
            updated_next_state = next_state['state']
        else:
            updated_next_state = []
            for l in next_state:
                updated_next_state.extend(l)

        return updated_next_state, reward, done, info

    def sample_action(self):
        return super().sample_action()