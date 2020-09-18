from types import SimpleNamespace

HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        # 'env_name':         "PongDeterministic-v4",
        'stop_mean_episode_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   100000,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'train_freq':       4,
    }),
    'breakout': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        # 'env_name':         "BreakoutDeterministic-v4",
        'stop_mean_episode_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      1000000,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   1000000,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32,
        'train_freq':       4,
    }),
    'invaders': SimpleNamespace(**{
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_mean_episode_reward': 500.0,
        'run_name': 'breakout',
        'replay_size':      1000000,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   1000000,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32,
        'train_freq':       4,
    }),
}





