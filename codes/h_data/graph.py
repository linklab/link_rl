import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

ddpg1 = pd.read_csv('matlab_ddbg_episode_reward_1.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])
ddpg2 = pd.read_csv('matlab_ddbg_episode_reward_2.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])
ddpg3 = pd.read_csv('matlab_ddbg_episode_reward_3.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])
ddpg4 = pd.read_csv('matlab_ddbg_episode_reward_4.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])
ddpg5 = pd.read_csv('matlab_ddbg_episode_reward_5.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])

ddpg1_reward = ddpg1['reward']
ddpg2_reward = ddpg2['reward']
ddpg3_reward = ddpg3['reward']
ddpg4_reward = ddpg4['reward']
ddpg5_reward = ddpg5['reward']

ddpg1_step = ddpg1['step_idx']
ddpg2_step = ddpg2['step_idx']
ddpg3_step = ddpg3['step_idx']
ddpg4_step = ddpg4['step_idx']
ddpg5_step = ddpg5['step_idx']

ddpg1_step_list = ddpg1_step.to_list()
ddpg2_step_list = ddpg2_step.to_list()
ddpg3_step_list = ddpg3_step.to_list()
ddpg4_step_list = ddpg4_step.to_list()
ddpg5_step_list = ddpg5_step.to_list()


ddpg1_reward_list = ddpg1_reward.to_list()
ddpg2_reward_list = ddpg2_reward.to_list()
ddpg3_reward_list = ddpg3_reward.to_list()
ddpg4_reward_list = ddpg4_reward.to_list()
ddpg5_reward_list = ddpg5_reward.to_list()

ddpg1_reward_list_step = []
for i in range(len(ddpg1_reward_list)):
    ddpg1_reward_list_step.extend([float(ddpg1_reward_list[i]) for _ in range(int(ddpg1_step_list[i]))])

print(len(ddpg1_reward_list_step))

td31 = pd.read_csv('matlab_td3_episode_reward_average_1.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])
td32 = pd.read_csv('matlab_td3_episode_reward_average_2.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])
td33 = pd.read_csv('matlab_td3_episode_reward_average_3.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])
td34 = pd.read_csv('matlab_td3_episode_reward_average_4.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])
td35 = pd.read_csv('matlab_td3_episode_reward_average_5.csv', names=['step_idx', 'epi_idx', 'epi_idx1', 'epi_idx2', 'reward', 'reward1', 'reward2'])

td31_reward = td31['reward']
td32_reward = td32['reward']
td33_reward = td33['reward']
td34_reward = td34['reward']
td35_reward = td35['reward']

td31_step = td31['step_idx']
td32_step = td32['step_idx']
td33_step = td33['step_idx']
td34_step = td34['step_idx']
td35_step = td35['step_idx']

td31_step_list = td31_step.to_list()
td32_step_list = td32_step.to_list()
td33_step_list = td33_step.to_list()
td34_step_list = td34_step.to_list()
td35_step_list = td35_step.to_list()


td31_reward_list = td31_reward.to_list()
td32_reward_list = td32_reward.to_list()
td33_reward_list = td33_reward.to_list()
td34_reward_list = td34_reward.to_list()
td35_reward_list = td35_reward.to_list()

ddpg_mean_list = []
ddpg_max_list = []
ddpg_min_list = []
for i in range(600):
    ddpg = [float(ddpg1_reward_list[i]),float(ddpg2_reward_list[i]),float(ddpg3_reward_list[i]),float(ddpg4_reward_list[i]),float(ddpg5_reward_list[i])]
    ddpg_max = max(ddpg)
    ddpg_sorted = sorted(ddpg)
    ddpg_min = ddpg_sorted[2]
    mean = sum(ddpg) / len(ddpg)
    ddpg_mean_list.append(mean)
    ddpg_max_list.append(ddpg_max)
    ddpg_min_list.append(ddpg_min)

td3_mean_list = []
td3_max_list = []
td3_min_list = []
for i in range(600):
    td3 = [float(td31_reward_list[i]),float(td32_reward_list[i]),float(td33_reward_list[i]),float(td34_reward_list[i]),float(td35_reward_list[i])]
    td3_max = max(td3)
    td3_sorted = sorted(td3)
    td3_min = td3_sorted[2]
    mean = sum(td3) / len(td3)
    td3_mean_list.append(mean)
    td3_max_list.append(td3_max)
    td3_min_list.append(td3_min)


x = [i for i in range(1,601)]
x_new = np.linspace(1, 600, 50)
ddpg_mean_list_smooth = scipy.interpolate.make_interp_spline(x, ddpg_mean_list)
ddpg_max_list_smooth = scipy.interpolate.make_interp_spline(x, ddpg_max_list)
ddpg_min_list_smooth = scipy.interpolate.make_interp_spline(x, ddpg_min_list)
ddpg_mean_new = ddpg_mean_list_smooth(x_new)
ddpg_max_new = ddpg_max_list_smooth(x_new)
ddpg_min_new = ddpg_min_list_smooth(x_new)
plt.plot(x_new,ddpg_mean_new)
plt.plot(x_new,ddpg_max_new)
plt.plot(x_new,ddpg_min_new)
plt.xlabel("episode")
plt.ylabel('reward')
plt.legend(['mean', 'max', 'min'],loc = 2)
plt.ylim([0, 80000])
plt.title('DDPG_reward')
plt.fill_between(x_new, ddpg_min_new, ddpg_max_new, color='lightgray', alpha=0.5)
plt.show()

td3_mean_list_smooth = scipy.interpolate.make_interp_spline(x, td3_mean_list)
td3_max_list_smooth = scipy.interpolate.make_interp_spline(x, td3_max_list)
td3_min_list_smooth = scipy.interpolate.make_interp_spline(x, td3_min_list)
td3_mean_new = td3_mean_list_smooth(x_new)
td3_max_new = td3_max_list_smooth(x_new)
td3_min_new = td3_min_list_smooth(x_new)
plt.plot(x_new,td3_mean_new)
plt.plot(x_new,td3_max_new)
plt.plot(x_new,td3_min_new)
plt.xlabel("episode")
plt.ylabel('reward')
plt.legend(['mean', 'max', 'min'])
plt.ylim([0, 80000])
plt.title('TD3_reward')
plt.fill_between(x_new, td3_min_new, td3_max_new, color='lightgray', alpha=0.5)
plt.show()