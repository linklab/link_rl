import gym
from gym import envs
import pybullet_envs
from gym.spaces import Discrete, Box

for idx, env_spec in enumerate(pybullet_envs.registry.all()):
    # if idx in range(0, 967): # GYM
    #     continue

    if idx in range(968, 1057): # MUJOCO
        continue

    # 1057 ~ : pybullet


    env = gym.make(env_spec.id)
    observation_space = env.observation_space
    action_space = env.action_space

    # if isinstance(action_space, Discrete):
    #     env.close()
    #     continue

    observation_space_str = "OBS_SPACE: {0}, SHAPE: {1}".format(
        type(observation_space), observation_space.shape
    )

    action_space_str = "ACTION_SPACE: {0}, SHAPE: {1}".format(
        type(action_space), action_space.shape
    )
    if isinstance(action_space, Discrete):
        action_space_str += ", N: {0}".format(action_space.n)
    elif isinstance(action_space, Box):
        action_space_str += ", RANGE: {0}".format(action_space)

    print("{0:>4}: {1:35} | reward_threshold: {2} | {3:65} {4}".format(
        idx, env_spec.id, env_spec.reward_threshold,
        observation_space_str, action_space_str
    ))
    env.close()



# 0: ALE/Tetris-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 1: ALE/Tetris-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 2: ALE/Asterix-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 3: ALE/Asterix-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 4: ALE/Asteroids-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 5: ALE/Asteroids-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 6: ALE/MsPacman-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 7: ALE/MsPacman-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 8: ALE/Kaboom-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 9: ALE/Kaboom-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 10: ALE/BankHeist-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 11: ALE/BankHeist-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 12: ALE/Backgammon-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 13: ALE/Backgammon-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 14: ALE/Klax-v5                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 15: ALE/Klax-ram-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 16: ALE/Crossbow-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 17: ALE/Crossbow-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 18: ALE/Kangaroo-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 19: ALE/Kangaroo-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 20: ALE/Skiing-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 21: ALE/Skiing-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 22: ALE/FishingDerby-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 23: ALE/FishingDerby-ram-v5             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 24: ALE/Krull-v5                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 25: ALE/Krull-ram-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 26: ALE/FlagCapture-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 27: ALE/FlagCapture-ram-v5              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 28: ALE/BasicMath-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 29: ALE/BasicMath-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 30: ALE/Berzerk-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 31: ALE/Berzerk-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 32: ALE/Tutankham-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 33: ALE/Tutankham-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 34: ALE/MarioBros-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 35: ALE/MarioBros-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 36: ALE/Zaxxon-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 37: ALE/Zaxxon-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 38: ALE/Venture-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 39: ALE/Venture-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 40: ALE/Riverraid-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 41: ALE/Riverraid-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 42: ALE/Centipede-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 43: ALE/Centipede-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 44: ALE/WordZapper-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 45: ALE/WordZapper-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 46: ALE/Adventure-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 47: ALE/Adventure-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 48: ALE/BeamRider-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 49: ALE/BeamRider-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 50: ALE/CrazyClimber-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 51: ALE/CrazyClimber-ram-v5             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 52: ALE/TimePilot-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 53: ALE/TimePilot-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 54: ALE/Carnival-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (214, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 55: ALE/Carnival-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 56: ALE/Tennis-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 57: ALE/Tennis-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 58: ALE/Seaquest-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 59: ALE/Seaquest-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 60: ALE/Bowling-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 61: ALE/Bowling-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 62: ALE/SpaceInvaders-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 63: ALE/SpaceInvaders-ram-v5            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 64: ALE/Pitfall2-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 65: ALE/Pitfall2-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 66: ALE/Freeway-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 67: ALE/Freeway-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 68: ALE/YarsRevenge-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 69: ALE/YarsRevenge-ram-v5              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 70: ALE/Casino-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 71: ALE/Casino-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 72: ALE/RoadRunner-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 73: ALE/RoadRunner-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 74: ALE/MiniatureGolf-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 75: ALE/MiniatureGolf-ram-v5            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 76: ALE/JourneyEscape-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (230, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 77: ALE/JourneyEscape-ram-v5            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 78: ALE/WizardOfWor-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 79: ALE/WizardOfWor-ram-v5              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 80: ALE/DonkeyKong-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 81: ALE/DonkeyKong-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 82: ALE/Gopher-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 83: ALE/Gopher-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 84: ALE/Breakout-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 85: ALE/Breakout-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 86: ALE/StarGunner-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 87: ALE/StarGunner-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 88: ALE/Othello-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 89: ALE/Othello-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 90: ALE/Atlantis-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 91: ALE/Atlantis-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 92: ALE/TicTacToe3D-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 93: ALE/TicTacToe3D-ram-v5              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 94: ALE/DoubleDunk-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 95: ALE/DoubleDunk-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 96: ALE/Hero-v5                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 97: ALE/Hero-ram-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 98: ALE/BattleZone-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 99: ALE/BattleZone-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 100: ALE/KeystoneKapers-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 101: ALE/KeystoneKapers-ram-v5           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 102: ALE/Solaris-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 103: ALE/Solaris-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 104: ALE/UpNDown-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 105: ALE/UpNDown-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 106: ALE/Frostbite-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 107: ALE/Frostbite-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 108: ALE/VideoCheckers-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 109: ALE/VideoCheckers-ram-v5            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 110: ALE/KungFuMaster-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 111: ALE/KungFuMaster-ram-v5             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 112: ALE/Trondead-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 113: ALE/Trondead-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 114: ALE/Earthworld-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 115: ALE/Earthworld-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 116: ALE/Koolaid-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 117: ALE/Koolaid-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 118: ALE/Pooyan-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (220, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 119: ALE/Pooyan-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 120: ALE/Pitfall-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 121: ALE/Pitfall-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 122: ALE/Turmoil-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 123: ALE/Turmoil-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 124: ALE/Videochess-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 125: ALE/Videochess-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 126: ALE/Entombed-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 127: ALE/Entombed-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 128: ALE/MontezumaRevenge-v5             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 129: ALE/MontezumaRevenge-ram-v5         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 130: ALE/PrivateEye-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 131: ALE/PrivateEye-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 132: ALE/Surround-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 133: ALE/Surround-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 134: ALE/AirRaid-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 135: ALE/AirRaid-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 136: ALE/Amidar-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 137: ALE/Amidar-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 138: ALE/SpaceWar-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 139: ALE/SpaceWar-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 140: ALE/Pacman-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 141: ALE/Pacman-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 142: ALE/Robotank-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 143: ALE/Robotank-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 144: ALE/LostLuggage-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 145: ALE/LostLuggage-ram-v5              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 146: ALE/DemonAttack-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 147: ALE/DemonAttack-ram-v5              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 148: ALE/Defender-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 149: ALE/Defender-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 150: ALE/NameThisGame-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 151: ALE/NameThisGame-ram-v5             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 152: ALE/Phoenix-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 153: ALE/Phoenix-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 154: ALE/Gravitar-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 155: ALE/Gravitar-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 156: ALE/Atlantis2-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 157: ALE/Atlantis2-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 158: ALE/LaserGates-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 159: ALE/LaserGates-ram-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 160: ALE/ElevatorAction-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 161: ALE/ElevatorAction-ram-v5           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 162: ALE/Pong-v5                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 163: ALE/Pong-ram-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 164: ALE/Hangman-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 165: ALE/Hangman-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 166: ALE/SirLancelot-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 167: ALE/SirLancelot-ram-v5              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 168: ALE/VideoPinball-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 169: ALE/VideoPinball-ram-v5             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 170: ALE/IceHockey-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 171: ALE/IceHockey-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 172: ALE/Boxing-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 173: ALE/Boxing-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 174: ALE/HauntedHouse-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 175: ALE/HauntedHouse-ram-v5             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 176: ALE/Assault-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 177: ALE/Assault-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 178: ALE/Alien-v5                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 179: ALE/Alien-ram-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 180: ALE/Qbert-v5                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 181: ALE/Qbert-ram-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 182: ALE/Enduro-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 183: ALE/Enduro-ram-v5                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 184: ALE/Videocube-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 185: ALE/Videocube-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 186: ALE/Et-v5                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 187: ALE/Et-ram-v5                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 188: ALE/KingKong-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 189: ALE/KingKong-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 190: ALE/MrDo-v5                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 191: ALE/MrDo-ram-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 192: ALE/Blackjack-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 193: ALE/Blackjack-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 194: ALE/ChopperCommand-v5               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 195: ALE/ChopperCommand-ram-v5           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 196: ALE/Galaxian-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 197: ALE/Galaxian-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 198: ALE/Frogger-v5                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 199: ALE/Frogger-ram-v5                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 200: ALE/Darkchambers-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 201: ALE/Darkchambers-ram-v5             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 202: ALE/Jamesbond-v5                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 203: ALE/Jamesbond-ram-v5                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 204: ALE/HumanCannonball-v5              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 205: ALE/HumanCannonball-ram-v5          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 206: ALE/Superman-v5                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 207: ALE/Superman-ram-v5                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 208: Adventure-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 209: AdventureDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 210: AdventureNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 211: Adventure-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 212: AdventureDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 213: AdventureNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 214: Adventure-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 215: Adventure-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 216: Adventure-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 217: Adventure-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 218: Adventure-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 219: Adventure-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 220: AirRaid-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 221: AirRaidDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 222: AirRaidNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 223: AirRaid-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 224: AirRaidDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 225: AirRaidNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (250, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 226: AirRaid-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 227: AirRaid-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 228: AirRaid-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 229: AirRaid-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 230: AirRaid-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 231: AirRaid-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 232: Alien-v0                            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 233: AlienDeterministic-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 234: AlienNoFrameskip-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 235: Alien-v4                            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 236: AlienDeterministic-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 237: AlienNoFrameskip-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 238: Alien-ram-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 239: Alien-ramDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 240: Alien-ramNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 241: Alien-ram-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 242: Alien-ramDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 243: Alien-ramNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 244: Amidar-v0                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 245: AmidarDeterministic-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 246: AmidarNoFrameskip-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 247: Amidar-v4                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 248: AmidarDeterministic-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 249: AmidarNoFrameskip-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 250: Amidar-ram-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 251: Amidar-ramDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 252: Amidar-ramNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 253: Amidar-ram-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 254: Amidar-ramDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 255: Amidar-ramNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 256: Assault-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 257: AssaultDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 258: AssaultNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 259: Assault-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 260: AssaultDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 261: AssaultNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 262: Assault-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 263: Assault-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 264: Assault-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 265: Assault-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 266: Assault-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 267: Assault-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 7
# 268: Asterix-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 269: AsterixDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 270: AsterixNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 271: Asterix-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 272: AsterixDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 273: AsterixNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 274: Asterix-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 275: Asterix-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 276: Asterix-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 277: Asterix-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 278: Asterix-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 279: Asterix-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 280: Asteroids-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 281: AsteroidsDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 282: AsteroidsNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 283: Asteroids-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 284: AsteroidsDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 285: AsteroidsNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 286: Asteroids-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 287: Asteroids-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 288: Asteroids-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 289: Asteroids-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 290: Asteroids-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 291: Asteroids-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 292: Atlantis-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 293: AtlantisDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 294: AtlantisNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 295: Atlantis-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 296: AtlantisDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 297: AtlantisNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 298: Atlantis-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 299: Atlantis-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 300: Atlantis-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 301: Atlantis-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 302: Atlantis-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 303: Atlantis-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 304: BankHeist-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 305: BankHeistDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 306: BankHeistNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 307: BankHeist-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 308: BankHeistDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 309: BankHeistNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 310: BankHeist-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 311: BankHeist-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 312: BankHeist-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 313: BankHeist-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 314: BankHeist-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 315: BankHeist-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 316: BattleZone-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 317: BattleZoneDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 318: BattleZoneNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 319: BattleZone-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 320: BattleZoneDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 321: BattleZoneNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 322: BattleZone-ram-v0                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 323: BattleZone-ramDeterministic-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 324: BattleZone-ramNoFrameskip-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 325: BattleZone-ram-v4                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 326: BattleZone-ramDeterministic-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 327: BattleZone-ramNoFrameskip-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 328: BeamRider-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 329: BeamRiderDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 330: BeamRiderNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 331: BeamRider-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 332: BeamRiderDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 333: BeamRiderNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 334: BeamRider-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 335: BeamRider-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 336: BeamRider-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 337: BeamRider-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 338: BeamRider-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 339: BeamRider-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 340: Berzerk-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 341: BerzerkDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 342: BerzerkNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 343: Berzerk-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 344: BerzerkDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 345: BerzerkNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 346: Berzerk-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 347: Berzerk-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 348: Berzerk-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 349: Berzerk-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 350: Berzerk-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 351: Berzerk-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 352: Bowling-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 353: BowlingDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 354: BowlingNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 355: Bowling-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 356: BowlingDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 357: BowlingNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 358: Bowling-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 359: Bowling-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 360: Bowling-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 361: Bowling-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 362: Bowling-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 363: Bowling-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 364: Boxing-v0                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 365: BoxingDeterministic-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 366: BoxingNoFrameskip-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 367: Boxing-v4                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 368: BoxingDeterministic-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 369: BoxingNoFrameskip-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 370: Boxing-ram-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 371: Boxing-ramDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 372: Boxing-ramNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 373: Boxing-ram-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 374: Boxing-ramDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 375: Boxing-ramNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 376: Breakout-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 377: BreakoutDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 378: BreakoutNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 379: Breakout-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 380: BreakoutDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 381: BreakoutNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 382: Breakout-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 383: Breakout-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 384: Breakout-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 385: Breakout-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 386: Breakout-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 387: Breakout-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 388: Carnival-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (214, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 389: CarnivalDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (214, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 390: CarnivalNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (214, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 391: Carnival-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (214, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 392: CarnivalDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (214, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 393: CarnivalNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (214, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 394: Carnival-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 395: Carnival-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 396: Carnival-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 397: Carnival-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 398: Carnival-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 399: Carnival-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 400: Centipede-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 401: CentipedeDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 402: CentipedeNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 403: Centipede-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 404: CentipedeDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 405: CentipedeNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 406: Centipede-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 407: Centipede-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 408: Centipede-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 409: Centipede-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 410: Centipede-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 411: Centipede-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 412: ChopperCommand-v0                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 413: ChopperCommandDeterministic-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 414: ChopperCommandNoFrameskip-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 415: ChopperCommand-v4                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 416: ChopperCommandDeterministic-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 417: ChopperCommandNoFrameskip-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 418: ChopperCommand-ram-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 419: ChopperCommand-ramDeterministic-v0  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 420: ChopperCommand-ramNoFrameskip-v0    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 421: ChopperCommand-ram-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 422: ChopperCommand-ramDeterministic-v4  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 423: ChopperCommand-ramNoFrameskip-v4    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 424: CrazyClimber-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 425: CrazyClimberDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 426: CrazyClimberNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 427: CrazyClimber-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 428: CrazyClimberDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 429: CrazyClimberNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 430: CrazyClimber-ram-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 431: CrazyClimber-ramDeterministic-v0    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 432: CrazyClimber-ramNoFrameskip-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 433: CrazyClimber-ram-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 434: CrazyClimber-ramDeterministic-v4    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 435: CrazyClimber-ramNoFrameskip-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 436: Defender-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 437: DefenderDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 438: DefenderNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 439: Defender-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 440: DefenderDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 441: DefenderNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 442: Defender-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 443: Defender-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 444: Defender-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 445: Defender-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 446: Defender-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 447: Defender-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 448: DemonAttack-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 449: DemonAttackDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 450: DemonAttackNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 451: DemonAttack-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 452: DemonAttackDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 453: DemonAttackNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 454: DemonAttack-ram-v0                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 455: DemonAttack-ramDeterministic-v0     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 456: DemonAttack-ramNoFrameskip-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 457: DemonAttack-ram-v4                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 458: DemonAttack-ramDeterministic-v4     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 459: DemonAttack-ramNoFrameskip-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 460: DoubleDunk-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 461: DoubleDunkDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 462: DoubleDunkNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 463: DoubleDunk-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 464: DoubleDunkDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 465: DoubleDunkNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 466: DoubleDunk-ram-v0                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 467: DoubleDunk-ramDeterministic-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 468: DoubleDunk-ramNoFrameskip-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 469: DoubleDunk-ram-v4                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 470: DoubleDunk-ramDeterministic-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 471: DoubleDunk-ramNoFrameskip-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 472: ElevatorAction-v0                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 473: ElevatorActionDeterministic-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 474: ElevatorActionNoFrameskip-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 475: ElevatorAction-v4                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 476: ElevatorActionDeterministic-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 477: ElevatorActionNoFrameskip-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 478: ElevatorAction-ram-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 479: ElevatorAction-ramDeterministic-v0  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 480: ElevatorAction-ramNoFrameskip-v0    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 481: ElevatorAction-ram-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 482: ElevatorAction-ramDeterministic-v4  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 483: ElevatorAction-ramNoFrameskip-v4    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 484: Enduro-v0                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 485: EnduroDeterministic-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 486: EnduroNoFrameskip-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 487: Enduro-v4                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 488: EnduroDeterministic-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 489: EnduroNoFrameskip-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 490: Enduro-ram-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 491: Enduro-ramDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 492: Enduro-ramNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 493: Enduro-ram-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 494: Enduro-ramDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 495: Enduro-ramNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 496: FishingDerby-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 497: FishingDerbyDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 498: FishingDerbyNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 499: FishingDerby-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 500: FishingDerbyDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 501: FishingDerbyNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 502: FishingDerby-ram-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 503: FishingDerby-ramDeterministic-v0    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 504: FishingDerby-ramNoFrameskip-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 505: FishingDerby-ram-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 506: FishingDerby-ramDeterministic-v4    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 507: FishingDerby-ramNoFrameskip-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 508: Freeway-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 509: FreewayDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 510: FreewayNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 511: Freeway-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 512: FreewayDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 513: FreewayNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 514: Freeway-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 515: Freeway-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 516: Freeway-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 517: Freeway-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 518: Freeway-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 519: Freeway-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 520: Frostbite-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 521: FrostbiteDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 522: FrostbiteNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 523: Frostbite-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 524: FrostbiteDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 525: FrostbiteNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 526: Frostbite-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 527: Frostbite-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 528: Frostbite-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 529: Frostbite-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 530: Frostbite-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 531: Frostbite-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 532: Gopher-v0                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 533: GopherDeterministic-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 534: GopherNoFrameskip-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 535: Gopher-v4                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 536: GopherDeterministic-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 537: GopherNoFrameskip-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 538: Gopher-ram-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 539: Gopher-ramDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 540: Gopher-ramNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 541: Gopher-ram-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 542: Gopher-ramDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 543: Gopher-ramNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 544: Gravitar-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 545: GravitarDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 546: GravitarNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 547: Gravitar-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 548: GravitarDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 549: GravitarNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 550: Gravitar-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 551: Gravitar-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 552: Gravitar-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 553: Gravitar-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 554: Gravitar-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 555: Gravitar-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 556: Hero-v0                             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 557: HeroDeterministic-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 558: HeroNoFrameskip-v0                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 559: Hero-v4                             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 560: HeroDeterministic-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 561: HeroNoFrameskip-v4                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 562: Hero-ram-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 563: Hero-ramDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 564: Hero-ramNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 565: Hero-ram-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 566: Hero-ramDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 567: Hero-ramNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 568: IceHockey-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 569: IceHockeyDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 570: IceHockeyNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 571: IceHockey-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 572: IceHockeyDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 573: IceHockeyNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 574: IceHockey-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 575: IceHockey-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 576: IceHockey-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 577: IceHockey-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 578: IceHockey-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 579: IceHockey-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 580: Jamesbond-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 581: JamesbondDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 582: JamesbondNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 583: Jamesbond-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 584: JamesbondDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 585: JamesbondNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 586: Jamesbond-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 587: Jamesbond-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 588: Jamesbond-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 589: Jamesbond-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 590: Jamesbond-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 591: Jamesbond-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 592: JourneyEscape-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (230, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 593: JourneyEscapeDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (230, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 594: JourneyEscapeNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (230, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 595: JourneyEscape-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (230, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 596: JourneyEscapeDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (230, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 597: JourneyEscapeNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (230, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 598: JourneyEscape-ram-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 599: JourneyEscape-ramDeterministic-v0   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 600: JourneyEscape-ramNoFrameskip-v0     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 601: JourneyEscape-ram-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 602: JourneyEscape-ramDeterministic-v4   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 603: JourneyEscape-ramNoFrameskip-v4     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 16
# 604: Kangaroo-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 605: KangarooDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 606: KangarooNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 607: Kangaroo-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 608: KangarooDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 609: KangarooNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 610: Kangaroo-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 611: Kangaroo-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 612: Kangaroo-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 613: Kangaroo-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 614: Kangaroo-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 615: Kangaroo-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 616: Krull-v0                            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 617: KrullDeterministic-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 618: KrullNoFrameskip-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 619: Krull-v4                            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 620: KrullDeterministic-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 621: KrullNoFrameskip-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 622: Krull-ram-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 623: Krull-ramDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 624: Krull-ramNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 625: Krull-ram-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 626: Krull-ramDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 627: Krull-ramNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 628: KungFuMaster-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 629: KungFuMasterDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 630: KungFuMasterNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 631: KungFuMaster-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 632: KungFuMasterDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 633: KungFuMasterNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 634: KungFuMaster-ram-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 635: KungFuMaster-ramDeterministic-v0    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 636: KungFuMaster-ramNoFrameskip-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 637: KungFuMaster-ram-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 638: KungFuMaster-ramDeterministic-v4    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 639: KungFuMaster-ramNoFrameskip-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 14
# 640: MontezumaRevenge-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 641: MontezumaRevengeDeterministic-v0    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 642: MontezumaRevengeNoFrameskip-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 643: MontezumaRevenge-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 644: MontezumaRevengeDeterministic-v4    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 645: MontezumaRevengeNoFrameskip-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 646: MontezumaRevenge-ram-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 647: MontezumaRevenge-ramDeterministic-v0 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 648: MontezumaRevenge-ramNoFrameskip-v0  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 649: MontezumaRevenge-ram-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 650: MontezumaRevenge-ramDeterministic-v4 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 651: MontezumaRevenge-ramNoFrameskip-v4  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 652: MsPacman-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 653: MsPacmanDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 654: MsPacmanNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 655: MsPacman-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 656: MsPacmanDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 657: MsPacmanNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 658: MsPacman-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 659: MsPacman-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 660: MsPacman-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 661: MsPacman-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 662: MsPacman-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 663: MsPacman-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 664: NameThisGame-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 665: NameThisGameDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 666: NameThisGameNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 667: NameThisGame-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 668: NameThisGameDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 669: NameThisGameNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 670: NameThisGame-ram-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 671: NameThisGame-ramDeterministic-v0    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 672: NameThisGame-ramNoFrameskip-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 673: NameThisGame-ram-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 674: NameThisGame-ramDeterministic-v4    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 675: NameThisGame-ramNoFrameskip-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 676: Phoenix-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 677: PhoenixDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 678: PhoenixNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 679: Phoenix-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 680: PhoenixDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 681: PhoenixNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 682: Phoenix-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 683: Phoenix-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 684: Phoenix-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 685: Phoenix-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 686: Phoenix-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 687: Phoenix-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 688: Pitfall-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 689: PitfallDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 690: PitfallNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 691: Pitfall-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 692: PitfallDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 693: PitfallNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 694: Pitfall-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 695: Pitfall-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 696: Pitfall-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 697: Pitfall-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 698: Pitfall-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 699: Pitfall-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 700: Pong-v0                             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 701: PongDeterministic-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 702: PongNoFrameskip-v0                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 703: Pong-v4                             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 704: PongDeterministic-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 705: PongNoFrameskip-v4                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 706: Pong-ram-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 707: Pong-ramDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 708: Pong-ramNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 709: Pong-ram-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 710: Pong-ramDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 711: Pong-ramNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 712: Pooyan-v0                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (220, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 713: PooyanDeterministic-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (220, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 714: PooyanNoFrameskip-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (220, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 715: Pooyan-v4                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (220, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 716: PooyanDeterministic-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (220, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 717: PooyanNoFrameskip-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (220, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 718: Pooyan-ram-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 719: Pooyan-ramDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 720: Pooyan-ramNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 721: Pooyan-ram-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 722: Pooyan-ramDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 723: Pooyan-ramNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 724: PrivateEye-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 725: PrivateEyeDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 726: PrivateEyeNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 727: PrivateEye-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 728: PrivateEyeDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 729: PrivateEyeNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 730: PrivateEye-ram-v0                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 731: PrivateEye-ramDeterministic-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 732: PrivateEye-ramNoFrameskip-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 733: PrivateEye-ram-v4                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 734: PrivateEye-ramDeterministic-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 735: PrivateEye-ramNoFrameskip-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 736: Qbert-v0                            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 737: QbertDeterministic-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 738: QbertNoFrameskip-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 739: Qbert-v4                            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 740: QbertDeterministic-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 741: QbertNoFrameskip-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 742: Qbert-ram-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 743: Qbert-ramDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 744: Qbert-ramNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 745: Qbert-ram-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 746: Qbert-ramDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 747: Qbert-ramNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 748: Riverraid-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 749: RiverraidDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 750: RiverraidNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 751: Riverraid-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 752: RiverraidDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 753: RiverraidNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 754: Riverraid-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 755: Riverraid-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 756: Riverraid-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 757: Riverraid-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 758: Riverraid-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 759: Riverraid-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 760: RoadRunner-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 761: RoadRunnerDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 762: RoadRunnerNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 763: RoadRunner-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 764: RoadRunnerDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 765: RoadRunnerNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 766: RoadRunner-ram-v0                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 767: RoadRunner-ramDeterministic-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 768: RoadRunner-ramNoFrameskip-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 769: RoadRunner-ram-v4                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 770: RoadRunner-ramDeterministic-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 771: RoadRunner-ramNoFrameskip-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 772: Robotank-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 773: RobotankDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 774: RobotankNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 775: Robotank-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 776: RobotankDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 777: RobotankNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 778: Robotank-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 779: Robotank-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 780: Robotank-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 781: Robotank-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 782: Robotank-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 783: Robotank-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 784: Seaquest-v0                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 785: SeaquestDeterministic-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 786: SeaquestNoFrameskip-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 787: Seaquest-v4                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 788: SeaquestDeterministic-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 789: SeaquestNoFrameskip-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 790: Seaquest-ram-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 791: Seaquest-ramDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 792: Seaquest-ramNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 793: Seaquest-ram-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 794: Seaquest-ramDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 795: Seaquest-ramNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 796: Skiing-v0                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 797: SkiingDeterministic-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 798: SkiingNoFrameskip-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 799: Skiing-v4                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 800: SkiingDeterministic-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 801: SkiingNoFrameskip-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 802: Skiing-ram-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 803: Skiing-ramDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 804: Skiing-ramNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 805: Skiing-ram-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 806: Skiing-ramDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 807: Skiing-ramNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3
# 808: Solaris-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 809: SolarisDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 810: SolarisNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 811: Solaris-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 812: SolarisDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 813: SolarisNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 814: Solaris-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 815: Solaris-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 816: Solaris-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 817: Solaris-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 818: Solaris-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 819: Solaris-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 820: SpaceInvaders-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 821: SpaceInvadersDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 822: SpaceInvadersNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 823: SpaceInvaders-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 824: SpaceInvadersDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 825: SpaceInvadersNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 826: SpaceInvaders-ram-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 827: SpaceInvaders-ramDeterministic-v0   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 828: SpaceInvaders-ramNoFrameskip-v0     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 829: SpaceInvaders-ram-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 830: SpaceInvaders-ramDeterministic-v4   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 831: SpaceInvaders-ramNoFrameskip-v4     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 832: StarGunner-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 833: StarGunnerDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 834: StarGunnerNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 835: StarGunner-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 836: StarGunnerDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 837: StarGunnerNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 838: StarGunner-ram-v0                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 839: StarGunner-ramDeterministic-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 840: StarGunner-ramNoFrameskip-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 841: StarGunner-ram-v4                   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 842: StarGunner-ramDeterministic-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 843: StarGunner-ramNoFrameskip-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 844: Tennis-v0                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 845: TennisDeterministic-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 846: TennisNoFrameskip-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 847: Tennis-v4                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 848: TennisDeterministic-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 849: TennisNoFrameskip-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 850: Tennis-ram-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 851: Tennis-ramDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 852: Tennis-ramNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 853: Tennis-ram-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 854: Tennis-ramDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 855: Tennis-ramNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 856: TimePilot-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 857: TimePilotDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 858: TimePilotNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 859: TimePilot-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 860: TimePilotDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 861: TimePilotNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 862: TimePilot-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 863: TimePilot-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 864: TimePilot-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 865: TimePilot-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 866: TimePilot-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 867: TimePilot-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 868: Tutankham-v0                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 869: TutankhamDeterministic-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 870: TutankhamNoFrameskip-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 871: Tutankham-v4                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 872: TutankhamDeterministic-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 873: TutankhamNoFrameskip-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 874: Tutankham-ram-v0                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 875: Tutankham-ramDeterministic-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 876: Tutankham-ramNoFrameskip-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 877: Tutankham-ram-v4                    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 878: Tutankham-ramDeterministic-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 879: Tutankham-ramNoFrameskip-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 8
# 880: UpNDown-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 881: UpNDownDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 882: UpNDownNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 883: UpNDown-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 884: UpNDownDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 885: UpNDownNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 886: UpNDown-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 887: UpNDown-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 888: UpNDown-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 889: UpNDown-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 890: UpNDown-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 891: UpNDown-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 892: Venture-v0                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 893: VentureDeterministic-v0             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 894: VentureNoFrameskip-v0               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 895: Venture-v4                          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 896: VentureDeterministic-v4             | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 897: VentureNoFrameskip-v4               | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 898: Venture-ram-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 899: Venture-ramDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 900: Venture-ramNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 901: Venture-ram-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 902: Venture-ramDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 903: Venture-ramNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 904: VideoPinball-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 905: VideoPinballDeterministic-v0        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 906: VideoPinballNoFrameskip-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 907: VideoPinball-v4                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 908: VideoPinballDeterministic-v4        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 909: VideoPinballNoFrameskip-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 910: VideoPinball-ram-v0                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 911: VideoPinball-ramDeterministic-v0    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 912: VideoPinball-ramNoFrameskip-v0      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 913: VideoPinball-ram-v4                 | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 914: VideoPinball-ramDeterministic-v4    | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 915: VideoPinball-ramNoFrameskip-v4      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 9
# 916: WizardOfWor-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 917: WizardOfWorDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 918: WizardOfWorNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 919: WizardOfWor-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 920: WizardOfWorDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 921: WizardOfWorNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 922: WizardOfWor-ram-v0                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 923: WizardOfWor-ramDeterministic-v0     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 924: WizardOfWor-ramNoFrameskip-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 925: WizardOfWor-ram-v4                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 926: WizardOfWor-ramDeterministic-v4     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 927: WizardOfWor-ramNoFrameskip-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 10
# 928: YarsRevenge-v0                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 929: YarsRevengeDeterministic-v0         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 930: YarsRevengeNoFrameskip-v0           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 931: YarsRevenge-v4                      | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 932: YarsRevengeDeterministic-v4         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 933: YarsRevengeNoFrameskip-v4           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 934: YarsRevenge-ram-v0                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 935: YarsRevenge-ramDeterministic-v0     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 936: YarsRevenge-ramNoFrameskip-v0       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 937: YarsRevenge-ram-v4                  | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 938: YarsRevenge-ramDeterministic-v4     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 939: YarsRevenge-ramNoFrameskip-v4       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 940: Zaxxon-v0                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 941: ZaxxonDeterministic-v0              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 942: ZaxxonNoFrameskip-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 943: Zaxxon-v4                           | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 944: ZaxxonDeterministic-v4              | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 945: ZaxxonNoFrameskip-v4                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (210, 160, 3)     ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 946: Zaxxon-ram-v0                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 947: Zaxxon-ramDeterministic-v0          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 948: Zaxxon-ramNoFrameskip-v0            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 949: Zaxxon-ram-v4                       | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 950: Zaxxon-ramDeterministic-v4          | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 951: Zaxxon-ramNoFrameskip-v4            | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (128,)            ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 18
# 952: CartPole-v0                         | reward_threshold: 195.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,)              ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 2
# 953: CartPole-v1                         | reward_threshold: 475.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,)              ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 2
# 954: MountainCar-v0                      | reward_threshold: -110.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,)              ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 3

# 955: MountainCarContinuous-v0            | reward_threshold: 90.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (1,), RANGE: Box([-1.], [1.], (1,), float32)

# 956: Pendulum-v1                         | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (3,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (1,), RANGE: Box([-2.], [2.], (1,), float32)

# 959: LunarLanderContinuous-v2            | reward_threshold: 200 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,), RANGE: Box([-1. -1.], [1. 1.], (2,), float32)

# 960: BipedalWalker-v3                    | reward_threshold: 300 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (24,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,), RANGE: Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)

# 961: BipedalWalkerHardcore-v3            | reward_threshold: 300 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (24,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,), RANGE: Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)

# 962: CarRacing-v0                        | reward_threshold: 900 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (96, 96, 3)       ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (3,), RANGE: Box([-1.  0.  0.], [1. 1. 1.], (3,), float32)

# 963: Blackjack-v1                        | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.tuple.Tuple'>, SHAPE: None          ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 2
# 964: FrozenLake-v1                       | reward_threshold: 0.7 | OBS_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: ()      ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 965: FrozenLake8x8-v1                    | reward_threshold: 0.85 | OBS_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: ()      ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 966: CliffWalking-v0                     | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: ()      ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 4
# 967: Taxi-v3                             | reward_threshold: 8 | OBS_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: ()      ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 6
# 1057: CartPoleBulletEnv-v1                | reward_threshold: 190.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,)              ACTION_SPACE: <class 'gym.spaces.discrete.Discrete'>, SHAPE: (), N: 2

# 1058: CartPoleContinuousBulletEnv-v0      | reward_threshold: 190.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (1,), RANGE: Box([-10.], [10.], (1,), float32)


# 1059: MinitaurBulletEnv-v0                | reward_threshold: 15.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (28,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1. 1.], (8,), float32)

# 1060: MinitaurBulletDuckEnv-v0            | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (28,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1. 1.], (8,), float32)

# 1061: MinitaurExtendedEnv-v0              | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (20,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1. 1.], (8,), float32)

# 1062: MinitaurReactiveEnv-v0              | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (12,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5], [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5], (8,), float32)

# 1063: MinitaurBallGymEnv-v0               | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (1,), RANGE: Box([-1.], [1.], (1,), float32)

# 1064: MinitaurTrottingEnv-v0              | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([-0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25], [0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25], (8,), float32)

# 1065: MinitaurStandGymEnv-v0              | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (28,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (1,), RANGE: Box([-1.], [1.], (1,), float32)

# 1066: MinitaurAlternatingLegsEnv-v0       | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([-0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1], [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1], (8,), float32)

# 1067: MinitaurFourLegStandEnv-v0          | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (4,), RANGE: Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)

# 1068: RacecarBulletEnv-v0                 | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,), RANGE: Box([-1. -1.], [1. 1.], (2,), float32)

# 1069: RacecarZedBulletEnv-v0              | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (10, 100, 4)      ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,), RANGE: Box([-1. -1.], [1. 1.], (2,), float32)

# 1070: KukaBulletEnv-v0                    | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (3,), RANGE: Box([-1. -1. -1.], [1. 1. 1.], (3,), float32)
# 1071: KukaCamBulletEnv-v0                 | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (256, 341, 4)     ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (3,), RANGE: Box([-1. -1. -1.], [1. 1. 1.], (3,), float32)
# 1072: KukaDiverseObjectGrasping-v0        | reward_threshold: 5.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (48, 48, 3)       ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (3,), RANGE: Box([-1. -1. -1.], [1. 1. 1.], (3,), float32)
# 1073: InvertedPendulumBulletEnv-v0        | reward_threshold: 950.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (5,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (1,), RANGE: Box([-1.], [1.], (1,), float32)
# 1074: InvertedDoublePendulumBulletEnv-v0  | reward_threshold: 9100.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (1,), RANGE: Box([-1.], [1.], (1,), float32)
# 1075: InvertedPendulumSwingupBulletEnv-v0 | reward_threshold: 800.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (5,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (1,), RANGE: Box([-1.], [1.], (1,), float32)
# 1076: ReacherBulletEnv-v0                 | reward_threshold: 18.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (9,)              ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (2,), RANGE: Box([-1. -1.], [1. 1.], (2,), float32)
# 1077: PusherBulletEnv-v0                  | reward_threshold: 18.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (55,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (7,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.], (7,), float32)
# 1078: ThrowerBulletEnv-v0                 | reward_threshold: 18.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (48,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (7,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.], (7,), float32)

# 1079: Walker2DBulletEnv-v0                | reward_threshold: 2500.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (22,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (6,), RANGE: Box([-1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1.], (6,), float32)
# 1080: HalfCheetahBulletEnv-v0             | reward_threshold: 3000.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (26,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (6,), RANGE: Box([-1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1.], (6,), float32)
# 1081: AntBulletEnv-v0                     | reward_threshold: 2500.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (28,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (8,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1. 1.], (8,), float32)
# 1082: HopperBulletEnv-v0                  | reward_threshold: 2500.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (15,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (3,), RANGE: Box([-1. -1. -1.], [1. 1. 1.], (3,), float32)
# 1083: HumanoidBulletEnv-v0                | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (44,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (17,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.], (17,), float32)
# 1084: HumanoidFlagrunBulletEnv-v0         | reward_threshold: 2000.0 | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (44,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (17,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.], (17,), float32)
# 1085: HumanoidFlagrunHarderBulletEnv-v0   | reward_threshold: None | OBS_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (44,)             ACTION_SPACE: <class 'gym.spaces.box.Box'>, SHAPE: (17,), RANGE: Box([-1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.], (17,), float32)