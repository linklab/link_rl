import collections
import copy
import enum
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=np.inf)

# General Parameters
PRINT_INTERVAL = 20
TEST_INTERVAL = 1000

# Task Parameters
NUM_TASKS = 15
INITIAL_RESOURCES_CAPACITY = [80, 80]  # task resource limits
SUM_RESOURCE_CAPACITY = sum(INITIAL_RESOURCES_CAPACITY)
LOW_DEMAND_RESOURCE_AT_TASK = [1, 1]
SUM_LOW_DEMAND_CAPACITY = sum(LOW_DEMAND_RESOURCE_AT_TASK)
HIGH_DEMAND_RESOURCE_AT_TASK = [20, 20]
INITIAL_TASK_DISTRIBUTION_FIXED = True

# DQN Parameters
LEARNING_RATE = 0.001
GAMMA = 0.99
BUFFER_LIMIT = 10000
BATCH_SIZE = 32
Q_NET_SYNC_INTERVAL = 100
MIN_BUFFER_SIZE_FOR_TRAIN = 500
NUM_EPISODES = 20000

# Epsilon Decaying Parameters
EPSILON_START = 0.9
EPSILON_END = 0.005
EPSILON_LAST_EPISODES_RATIO = 0.5

VERBOSE = False
FIGURE = True


class DoneReasonType(enum.Enum):
    TYPE_1 = "Type 1"  # The Same Task Selected
    TYPE_2 = "Type 2"  # Resource Limit Exceeded
    TYPE_3 = "Type 3"  # Resource allocated fully - BEST!!!
    TYPE_4 = "Type 4"  # All Tasks Selected


class EnvironmentTaskScheduling:
    def __init__(self):
        self.internal_state = None
        self.actions_selected = None
        self.resource_of_all_tasks_selected = None
        self.fixed_initial_internal_state = None
        self.min_task_resource_demand = None

        if INITIAL_TASK_DISTRIBUTION_FIXED:
            self.fixed_initial_internal_state = self.make_initial_internal_state()
            print(self.fixed_initial_internal_state)
            print("min_task_resource_demand:", self.min_task_resource_demand)
            print("###########################################################")

    def make_initial_internal_state(self):
        state = np.zeros(shape=(NUM_TASKS + 1, 3), dtype=int)

        min_task_resource_demand = SUM_RESOURCE_CAPACITY

        for task_idx in range(NUM_TASKS):
            resource_demand = np.random.randint(
                low=LOW_DEMAND_RESOURCE_AT_TASK,
                high=HIGH_DEMAND_RESOURCE_AT_TASK,
                size=(1, 2)
            )
            if np.sum(resource_demand) < min_task_resource_demand:
                min_task_resource_demand = np.sum(resource_demand)

            state[task_idx][1:] = resource_demand

        self.min_task_resource_demand = min_task_resource_demand

        state[NUM_TASKS][1:] = np.array(INITIAL_RESOURCES_CAPACITY)

        return state

    def get_observation_from_internal_state(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / SUM_RESOURCE_CAPACITY
        return observation

    def reset(self):
        if INITIAL_TASK_DISTRIBUTION_FIXED:
            assert self.fixed_initial_internal_state is not None
            self.internal_state = copy.deepcopy(self.fixed_initial_internal_state)
            #print(self.internal_state)
        else:
            self.internal_state = self.make_initial_internal_state()

        # print(self.internal_state)

        self.actions_selected = []
        self.resource_of_all_tasks_selected = 0

        observation = self.get_observation_from_internal_state()

        return observation

    def get_sum(self, x):
        if x[0] == 1:
            return sum(x) - 1
        else:
            return 0

    def step(self, action_idx):
        info = {}
        self.actions_selected.append(action_idx)

        if self.internal_state[action_idx][0] == 1:
            resource_efficiency_reward = 0.0
            task_allocation_reward = 0.0
            misbehavior_reward = -1.0
            done = True
            info['DoneReasonType'] = DoneReasonType.TYPE_1    ##### [TYPE 1] The Same Task Selected #####

        else:
            self.internal_state[action_idx][0] = 1

            step_resource = self.internal_state[action_idx][1] + self.internal_state[action_idx][2]
            resource_of_all_tasks_selected_with_this_step = self.resource_of_all_tasks_selected + step_resource

            self.internal_state[action_idx][1] = -1
            self.internal_state[action_idx][2] = -1

            if resource_of_all_tasks_selected_with_this_step > SUM_RESOURCE_CAPACITY:
                resource_efficiency_reward = 0.0
                task_allocation_reward = 1.0 / NUM_TASKS
                misbehavior_reward = -1.0
                done = True
                info['DoneReasonType'] = DoneReasonType.TYPE_2  ##### [TYPE 2] Resource Limit Exceeded #####

            elif SUM_RESOURCE_CAPACITY - resource_of_all_tasks_selected_with_this_step <= self.min_task_resource_demand:
                self.resource_of_all_tasks_selected = resource_of_all_tasks_selected_with_this_step
                self.internal_state[-1][0] = self.resource_of_all_tasks_selected

                bonus_reward = SUM_RESOURCE_CAPACITY / 2.0
                resource_efficiency_reward = (step_resource + bonus_reward) / SUM_RESOURCE_CAPACITY
                task_allocation_reward = 1.0 / NUM_TASKS
                misbehavior_reward = 0.0
                done = True
                info['DoneReasonType'] = DoneReasonType.TYPE_3  ##### [TYPE 3] Resource allocated fully - BEST!!! #####

            else:
                self.resource_of_all_tasks_selected = resource_of_all_tasks_selected_with_this_step
                self.internal_state[-1][0] = self.resource_of_all_tasks_selected

                resource_efficiency_reward = step_resource / SUM_RESOURCE_CAPACITY
                task_allocation_reward = 1.0 / NUM_TASKS
                misbehavior_reward = 0.0
                if 0 not in self.internal_state[:, 0]:
                    done = True
                    info['DoneReasonType'] = DoneReasonType.TYPE_4  ##### [TYPE 4] All Tasks Selected #####
                else:
                    done = False  ##### It's Normal Step (Not done)

        observation = self.get_observation_from_internal_state()
        reward = resource_efficiency_reward + task_allocation_reward + misbehavior_reward

        info['Actions selected'] = self.actions_selected
        info['Resources allocated'] = self.resource_of_all_tasks_selected
        info['Limit'] = SUM_RESOURCE_CAPACITY

        return observation, reward, done, info


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)

    def put(self, transition):
        # state, action, reward, new_state, done_mask
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        states, actions, rewards, new_states, done_masks = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, new_state, done_mask = transition
            states.append(state)
            actions.append([action])
            rewards.append([reward])
            new_states.append(new_state)
            done_masks.append([done_mask])

        # print(states)
        # print(actions)
        # print(rewards)
        # print(new_states)
        # print(done_masks)
        # print("####################")
        return torch.tensor(np.array(states), dtype=torch.float), \
               torch.tensor(np.array(actions), dtype=torch.int64), \
               torch.tensor(np.array(rewards), dtype=torch.float), \
               torch.tensor(np.array(new_states), dtype=torch.float), \
               torch.tensor(np.array(done_masks), dtype=torch.float)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(((NUM_TASKS + 1) * 3), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, NUM_TASKS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

    def get_action(self, obs, epsilon):
        coin = random.random()

        if coin < epsilon:
            action = random.randint(0, (NUM_TASKS - 1))
        else:
            out = self.forward(obs)
            action = int(out.argmax(dim=-1).item())

        return action


def train(q_net, q_net_target, memory, optimizer):
    states, actions, rewards, new_states, done_masks = memory.sample(BATCH_SIZE)

    q_net_out = q_net.forward(states)
    q_value = q_net_out.gather(dim=-1, index=actions)

    with torch.no_grad():
        max_q_prime = q_net_target(new_states).max(dim=-1).values.unsqueeze(dim=-1)
        target_value = rewards + GAMMA * max_q_prime * done_masks

    #print(states.shape, actions.shape, rewards.shape, new_states.shape, done_masks.shape, q_value.shape, max_q_prime.shape, target_value.shape)

    assert q_value.shape == target_value.shape
    loss = F.smooth_l1_loss(q_value, target_value.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def get_epsilon(n_episode):
    epsilon_decaying_last_episode = NUM_EPISODES * EPSILON_LAST_EPISODES_RATIO
    # Epsilon Annealing
    epsilon = max(EPSILON_END, EPSILON_START * (epsilon_decaying_last_episode - n_episode) / epsilon_decaying_last_episode)
    return epsilon


def main():
    train_env = EnvironmentTaskScheduling()

    q_net = QNet()
    q_net_target = QNet()
    q_net_target.load_state_dict(q_net.state_dict())

    memory = ReplayBuffer()

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    total_step = 0
    last_loss_value = 0.0
    
    train_score_list = []
    train_resource_allocation_list = []
    train_avg_score_list = []
    train_avg_resource_allocation_list = []

    train_loss_list = []
    train_avg_loss_list = []

    test_avg_score_list = []
    test_avg_resource_allocation_list = []

    test_std_score_list = []
    test_episode_list = []

    if INITIAL_TASK_DISTRIBUTION_FIXED:
        avg_score, std_score, avg_resource_allocated = test_main(q_net, test_env=train_env)
    else:
        test_env = EnvironmentTaskScheduling()
        avg_score, std_score, avg_resource_allocated = test_main(q_net, test_env=test_env)

    test_avg_score_list.append(avg_score)
    test_std_score_list.append(std_score)
    test_avg_resource_allocation_list.append(avg_resource_allocated)
    test_episode_list.append(0)

    for n_episode in range(NUM_EPISODES):
        epsilon = get_epsilon(n_episode)

        observation = train_env.reset()  # initialize task_scheduling state

        done = False
        info = None
        episode_step = 0
        score = 0.0
        if VERBOSE:
            print("\n========================================================================================")

        while not done:
            if VERBOSE:
                print("[Episode: {0}/{1}, Step: {2}] {3} ".format(
                    n_episode, NUM_EPISODES, episode_step, observation
                ), end="")

            action = q_net.get_action(torch.from_numpy(observation).float(), epsilon)

            new_observation, reward, done, info = train_env.step(action)

            if VERBOSE:
                print("action: {0}, {1}, reward: {2}, done: {3}".format(action, new_observation, reward, done))

            done_mask = 0.0 if done else 1.0
            memory.put((observation, action, reward, new_observation, done_mask))
            episode_step = episode_step + 1
            total_step = total_step + 1
            score += reward

            observation = new_observation

            # TRAIN SHOULD BE DONE EVERY STEP
            if memory.size() > MIN_BUFFER_SIZE_FOR_TRAIN:
                last_loss_value = train(q_net, q_net_target, memory, optimizer)

            if total_step % Q_NET_SYNC_INTERVAL == 0 and total_step != 0:
                q_net_target.load_state_dict(q_net.state_dict())

            if total_step % TEST_INTERVAL == 0:
                if INITIAL_TASK_DISTRIBUTION_FIXED:
                    avg_score, std_score, avg_resource_allocated = test_main(q_net, test_env=train_env)
                else:
                    test_env = EnvironmentTaskScheduling()
                    avg_score, std_score, avg_resource_allocated = test_main(q_net, test_env=test_env)

                test_avg_score_list.append(avg_score)
                test_std_score_list.append(std_score)
                test_avg_resource_allocation_list.append(avg_resource_allocated)
                test_episode_list.append(n_episode)

        train_score_list.append(score)
        train_avg_score_list.append(np.mean(train_score_list[-100:]))

        train_resource_allocation_list.append(info["Resources allocated"])
        train_avg_resource_allocation_list.append(np.mean(train_resource_allocation_list[-100:]))

        train_loss_list.append(last_loss_value)
        train_avg_loss_list.append(np.mean(train_loss_list[-100:]))

        if n_episode % PRINT_INTERVAL == 0 and n_episode != 0:
            print("Epi.: {0:5}/{1}, Score: {2:5.2f}, Mean Score: {3:5.2f}, Total Step: {4:6}, "
                  "Episode Step: {5:2}, Buffer: {6:5}, Epsilon: {7:4.1f}%, Last Loss: {8:5.3f}, "
                  "Done: {9} "
                  "{10} "
                  "(Resource Allocated: {11:5}, Limit: {12}, Actions: {13})".format(
                n_episode, NUM_EPISODES, score, train_avg_score_list[-1], total_step, 
                episode_step, memory.size(), epsilon * 100, last_loss_value,
                info["DoneReasonType"].value,
                "[BEST!]" if info["DoneReasonType"] in (DoneReasonType.TYPE_3, DoneReasonType.TYPE_4) else "",
                info["Resources allocated"], info["Limit"], info["Actions selected"]
            ))

        if VERBOSE:
            print("[Epi.: {0}/{1}, Step: {2}] Episode Score {3}, EPISODE DONE - {4} "
                  "(Resource Allocated: {6:5}, Limit: {7}, Actions: {5})".format(
                n_episode, NUM_EPISODES, episode_step, score, info["DoneReasonType"].value,
                info["Resources allocated"], info["Limit"], info["Actions selected"]
            ))
            print("========================================================================================")

    if INITIAL_TASK_DISTRIBUTION_FIXED:
        avg_score, std_score, avg_resource_allocated = test_main(q_net, test_env=train_env)
    else:
        test_env = EnvironmentTaskScheduling()
        avg_score, std_score, avg_resource_allocated = test_main(q_net, test_env=test_env)

    test_avg_score_list.append(avg_score)
    test_std_score_list.append(std_score)
    test_avg_resource_allocation_list.append(avg_resource_allocated)
    test_episode_list.append(NUM_EPISODES)

    if FIGURE:
        draw_performance(
            train_score_list, train_avg_score_list,
            train_resource_allocation_list, train_avg_resource_allocation_list,
            train_loss_list, train_avg_loss_list,
            test_avg_score_list, test_avg_resource_allocation_list, test_std_score_list, test_episode_list
        )


def draw_performance(
        train_score_list, train_avg_score_list,
        train_resource_allocation_list, train_avg_resource_allocation_list,
        train_loss_list, train_avg_loss_list,
        test_avg_score_list, test_avg_resource_allocation_list, test_std_score_list, test_episode_list
):
    fig = plt.figure(figsize=(6, 15))
    gs = fig.add_gridspec(6, hspace=0.5)
    ax = gs.subplots(sharex=True, sharey=False)

    plt.xlabel("Number of Episode")

    ax[0].set_title("1) Score & 2) Mean Score over Recent 100 Episodes (Orange)")
    ax[0].plot(train_score_list)
    ax[0].plot(train_avg_score_list)

    ax[1].set_title("1) Resource Allocated & 2) Mean Resource Allocated over Recent 100 Episodes (Orange)")
    ax[1].plot(train_resource_allocation_list)
    ax[1].plot(train_avg_resource_allocation_list)

    ax[2].set_title("2) Loss & 2) Mean Loss over Recent 100 Episodes (Orange)")
    ax[2].plot(train_loss_list)
    ax[2].plot(train_avg_loss_list)

    ax[3].set_title("[Test] Average Score over 3 Episodes")
    ax[3].plot(test_episode_list, test_avg_score_list)

    ax[4].set_title("[Test] Average Resource Allocated over 3 Episodes")
    ax[4].plot(test_episode_list, test_avg_resource_allocation_list)

    ax[5].set_title("[Test] Score Std. over 3 Episodes")
    ax[5].plot(test_episode_list, test_std_score_list)

    plt.savefig('result_{0}.png'.format(NUM_TASKS))
    plt.show()


def test_main(q_net, test_env):
    score = 0.0
    NUM_EPISODES = 3

    test_score_list = []
    test_resource_allocation_list = []

    for n_episode in range(NUM_EPISODES):
        observation = test_env.reset()
        done = False
        info = None

        episode_step = 0
        while not done:
            action = q_net.get_action(torch.from_numpy(observation).float(), epsilon=0.0) # Only Greedy Action Selection
            new_observation, reward, done, info = test_env.step(action)
            episode_step = episode_step + 1
            score += reward
            observation = new_observation

        test_score_list.append(score)
        test_resource_allocation_list.append(info["Resources allocated"])

    return np.average(test_score_list), np.std(test_score_list), np.average(test_resource_allocation_list)


if __name__ == '__main__':
    main()