import numpy as np
from collections import deque

class dummy_env:
    def __init__(self):

        self.state_0 = 0.0
        self.state_1 = 1.0
        self.state_2 = 2.0
        self.state_3 = 3.0
        self.state_4 = 4.0

        self.state_deque=deque(maxlen = 30)
        self.step_length = 4

    def reset(self):
        self.current_state = self.state_0

        if self.step_length== -1:
            state = np.array(self.current_state)
        elif self.step_length >= 1:
            state = np.tile(self.current_state, (self.step_length, 1)) # state: (step_size, 4)
        else:
            raise ValueError()

        return state

    def step(self, action):
        if action == 0:
            if self.current_state == self.state_0:
                self.current_state = self.state_1
            elif self.current_state == self.state_1:
                self.current_state = self.state_3
            elif self.current_state == self.state_2:
                self.current_state = self.state_3
        else:
            if self.current_state == self.state_0:
                self.current_state = self.state_2
            elif self.current_state == self.state_1:
                self.current_state = self.state_4
            elif self.current_state == self.state_2:
                self.current_state = self.state_4

        self.state_deque.append(self.current_state)

        reward = self.current_state

        if self.current_state == self.state_3 or self.current_state == self.state_4:
            done = True
        else:
            done = False

        if self.step_length == -1:
            next_state = np.array(self.state_deque[-1])
        elif self.step_length >= 1:
            if len(self.state_deque) < self.step_length:
                next_state = list(self.state_deque)


                for _ in range(self.step_length - len(self.state_deque)):
                    next_state.insert(0, [0.0] * self.obs_size)
                next_state = np.array(next_state)
                print()
            else:
                next_state = np.array(
                    [
                        self.state_deque[-self.step_length + offset] for offset in range(self.step_length)
                    ]
                )
            # print(next_state.shape)
            # print(next_state)
        else:
            raise ValueError()

        info = [None]


        print(next_state)
        return next_state, reward, done,info