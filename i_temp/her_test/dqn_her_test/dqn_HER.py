import random
from log_utils import logger, mean_val
from copy import deepcopy
import torch
import torch.nn.functional as F
from collections import deque
import numpy as np
import copy


class HER:
    def __init__(self, N):
        self.buffer = None
        self.N = N

    def reset(self):
        self.buffer = deque()

    def keep(self, item):
        self.buffer.append(item)

    def size(self):
        return len(self.buffer)

    def get_her_trajectory(self):
        new_buffer = deepcopy(self.buffer)
        num = len(new_buffer)

        # 에피소드 마지막에 도착한 상태를 her_goal로 지정
        her_goal = self.buffer[-1][3][0:self.N]

        for i in range(num):
            new_buffer[-1 - i][0][self.N:] = her_goal
            new_buffer[-1 - i][2] = -1.0
            new_buffer[-1 - i][3][self.N:] = her_goal
            new_buffer[-1 - i][4] = False
            if np.sum(np.abs((new_buffer[-1 - i][3][self.N:] - her_goal))) == 0:
                new_buffer[-1 - i][2] = 0.0
                new_buffer[-1 - i][4] = True

        return new_buffer


class Policy(torch.nn.Module):
    def __init__(self, N, K):
        super(Policy, self).__init__()
        self.N = N
        self.K = K
        self.fc1 = torch.nn.Linear(self.N, 128)
        self.fc2 = torch.nn.Linear(128, self.K)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN_HER:
    def __init__(self, env, gamma, buffer_size, cuda_flag):
        self.env = env
        self.N = env.N
        self.cuda_flag = cuda_flag

        if self.cuda_flag:
            self.model = Policy(2 * self.N, self.N).cuda()
            self.target_model = copy.deepcopy(self.model).cuda()
        else:
            self.model = Policy(2 * self.N, self.N)
            self.target_model = copy.deepcopy(self.model)

        self.her = HER(self.N)
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0005)
        self.batch_size = 64
        self.epsilon = 0.1
        self.buffer_size = buffer_size
        self.step_counter = 0
        self.epsi_high = 0.9
        self.epsi_low = 0.05
        self.steps = 0
        self.count = 0
        self.decay = 200
        self.eps = self.epsi_high
        self.update_target_step = 1000
        self.log = logger()
        self.log.add_log('tot_return')
        self.log.add_log('avg_loss')
        self.log.add_log('final_dist')

        self.replay_buffer = deque(maxlen=buffer_size)

    def run_episode(self):
        self.her.reset()
        state = self.env.reset()
        sum_r = 0
        mean_loss = mean_val()
        min_dist = self.N

        for t in range(self.N):
            self.steps += 1
            self.eps = self.epsi_low + (self.epsi_high - self.epsi_low) * (np.exp(-1.0 * self.steps / self.decay))
            if self.cuda_flag:
                Q = self.model(state.cuda())
            else:
                Q = self.model(state)

            num = np.random.rand()
            if num < self.eps:
                action = torch.randint(0, Q.shape[1], (1,)).type(torch.LongTensor)
            else:
                action = torch.argmax(Q, dim=1)

            new_state, reward, done, dist = self.env.step(state, action.item())
            sum_r = sum_r + reward

            if dist < min_dist:
                min_dist = dist

            if t + 1 == self.N:
                done = True

            self.replay_buffer.append([
                deepcopy(state.squeeze(0).numpy()),
                deepcopy(action),
                deepcopy(reward),
                deepcopy(new_state.squeeze(0).numpy()),
                deepcopy(done)
            ])

            if done:
                self.her.keep([
                    state.squeeze(0).numpy(), action, reward, new_state. squeeze(0).numpy(), done
                ])

            self.her.keep([
                state.squeeze(0).numpy(), action, reward, new_state.squeeze(0).numpy(), done
            ])

            loss = self.update_model()

            mean_loss.append(loss)
            state = deepcopy(new_state)

            self.step_counter = self.step_counter + 1

            if self.step_counter > self.update_target_step:
                self.target_model.load_state_dict(self.model.state_dict())
                self.step_counter = 0
                print('updated target model')

        her_trajectory = self.her.get_her_trajectory()

        for transition in her_trajectory:
            self.replay_buffer.append(transition)

        self.log.add_item('tot_return', sum_r)
        self.log.add_item('avg_loss', mean_loss.get())
        self.log.add_item('final_dist', min_dist)

    def update_model(self):
        self.optimizer.zero_grad()
        num = len(self.replay_buffer)
        K = np.min([num, self.batch_size])
        samples = random.sample(self.replay_buffer, K)

        S0, A0, R1, S1, D1 = zip(*samples)
        S0 = torch.tensor(S0, dtype=torch.float)
        A0 = torch.tensor(A0, dtype=torch.long).view(K, -1)
        R1 = torch.tensor(R1, dtype=torch.float).view(K, -1)
        S1 = torch.tensor(S1, dtype=torch.float)
        D1 = torch.tensor(D1, dtype=torch.float)

        if self.cuda_flag:
            target_q = R1.squeeze().cuda() + self.gamma*self.target_model(S1.cuda()).max(dim=1)[0].detach()*(1 - D1.cuda())
            policy_q = self.model(S0.cuda()).gather(1, A0.cuda())
        else:
            target_q = R1.squeeze() + self.gamma*self.target_model(S1).max(dim=1)[0].detach()*(1 - D1)
            policy_q = self.model(S0).gather(1, A0)

        L = F.smooth_l1_loss(policy_q.squeeze(), target_q.squeeze())
        L.backward()
        self.optimizer.step()
        return L.detach().item()

    def run_epoch(self):
        self.run_episode()
        return self.log