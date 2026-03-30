import gymnasium as gym

env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")
obs, info = env.reset()

print("=" * 40)
print(f"Observation Space: {env.observation_space}")
print(f"Observation Shape: {env.observation_space.shape}")
print(f"Action Space: {env.action_space}")
print(f"Action Shape: {env.action_space.shape}")
print("=" * 40)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()