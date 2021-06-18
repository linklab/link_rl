from pettingzoo.butterfly import pistonball_v4
env = pistonball_v4.env()

print(env.observation_spaces)

print(env.action_spaces)

obs = env.reset()
print(env.num_agents)

print(env.agents)