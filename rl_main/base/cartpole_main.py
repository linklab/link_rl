from common.environments.gym.cartpole import CartPole_v0

env = CartPole_v0()

state = env.reset()

max_episodes = 100

for episode in range(1, max_episodes + 1):
    done = False

    while not done:
        action = agent.get_action()
        next_state, reward, done, info = env.step(action)

        state = next_state
