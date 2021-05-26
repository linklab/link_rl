import pickle
import matplotlib.pyplot as plt
from collections import deque

with open('episode_reward_list_1.txt', 'rb') as f:
    data = pickle.load(f)


x = [i + 1 for i in range(1500)]
y = data[:1500]

y_deque = deque(maxlen=50)
y_mean = []

for i in y:
    y_deque.append(i)
    if len(y_deque) == 50:
        y_mean.append(sum(y_deque)/50)

x_mean = [i + 50 for i in range(len(y_mean))]
print(x)
print(y)

plt.plot(x, y, x_mean, y_mean, 'r-')
plt.title("MATLAB DDPG EPISODE REWARD")
# plt.xticks([i+1 for i in x if i+1 % 100 == 0])
plt.xlabel('episode')
plt.xticks([i * 150 for i in range(11)])
plt.ylabel('episode reward')
plt.legend(['episode reward', 'mean 50 episode reward'])
plt.show()