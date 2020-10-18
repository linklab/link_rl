from or_gym.envs.classic_or import KnapsackEnv
import numpy as np

class CustomUnboundedKnapsackEnv(KnapsackEnv):
    def __init__(self, *args, **kwargs):
        super(CustomUnboundedKnapsackEnv, self).__init__(*args, **kwargs)

    def reset(self):
        state = super().reset()
        if self.mask:
            updated_state = state['state']
        else:
            updated_state = []
            for l in state:
                updated_state.extend(l)

        return updated_state

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if self.mask:
            updated_next_state = next_state['state']
        else:
            updated_next_state = []
            for l in next_state:
                updated_next_state.extend(l)

        return updated_next_state, reward, done, info

    def sample_action(self):
        return super().sample_action()