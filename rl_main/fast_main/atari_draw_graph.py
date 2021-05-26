import matplotlib.pyplot as plt
import numpy as np
import pickle

from config.parameters import PARAMETERS as params
from config.names import *

EXPERIMENTS_N_STEPS = [1, 3, 6, 'Omega']
MAX_RUNS = 3

EMA_WINDOW_SIZE = 20

# episode_rewards_across_steps[n_steps][max_runs][max_steps/save_period]


def draw_graph(episode_rewards_across_steps, q_loss_across_steps):
    plt.clf()

    colors = ['b', 'y', 'r', 'c', 'm', 'g', 'k']

    # f, (ax0, ax1) = plt.subplots(2, figsize=(20, 20))
    f, (ax0) = plt.subplots(1, figsize=(20, 10))

    for n_step_idx, n_step in enumerate(EXPERIMENTS_N_STEPS):
        if n_step == 'Omega':
            label = r"$\Omega$"
        elif n_step == 1:
            label = "1 step"
        else:
            label = "{0} steps".format(n_step)

        score_ma_steps = np.mean(episode_rewards_across_steps[n_step_idx], axis=0)
        std_score_ma_steps = np.std(episode_rewards_across_steps[n_step_idx], axis=0)# / 10

        # runs_avg_cumulative_rewards_across_steps = np.mean(episode_rewards_across_steps[n_step_idx], axis=0)
        # score_ma_steps = exp_moving_average(runs_avg_cumulative_rewards_across_steps, EMA_WINDOW_SIZE)
        # runs_std_cumulative_rewards_across_steps = np.std(episode_rewards_across_steps[n_step_idx], axis=0) / 10
        # std_score_ma_steps = exp_moving_average(runs_std_cumulative_rewards_across_steps, EMA_WINDOW_SIZE)
        ax0.plot(
            range(int(params.MAX_GLOBAL_STEP / params.DATA_SAVE_STEP_PERIOD)),
            score_ma_steps,
            label=label,
            linewidth=2.0 if n_step == 'Omega' else 1.0,
            linestyle='-' if n_step == 'Omega' else '--',
            color=colors[n_step_idx]
        )
        ax0.fill_between(
            range(int(params.MAX_GLOBAL_STEP / params.DATA_SAVE_STEP_PERIOD)),
            np.clip(score_ma_steps - std_score_ma_steps, env_episode_min_reward(params.ENVIRONMENT_ID.value), score_ma_steps - std_score_ma_steps),    # score_ma_steps - std_score_ma_steps
            score_ma_steps + std_score_ma_steps,
            alpha=0.1,
            color=colors[n_step_idx]
        )
        ax0.tick_params(labelsize=32)

        # loss_ma_steps = np.mean(q_loss_across_steps[n_step_idx], axis=0)
        # std_q_loss_ma_steps = np.std(q_loss_across_steps[n_step_idx], axis=0)# / 10
        #
        # # runs_avg_q_loss_across_steps = np.mean(q_loss_across_steps[n_step_idx], axis=0)
        # # loss_ma_steps = exp_moving_average(runs_avg_q_loss_across_steps, EMA_WINDOW_SIZE)
        # # runs_std_q_loss_across_steps = np.std(q_loss_across_steps[n_step_idx], axis=0) / 10
        # # std_q_loss_ma_steps = exp_moving_average(runs_std_q_loss_across_steps, EMA_WINDOW_SIZE)
        # ax1.plot(
        #     range(int(params.MAX_GLOBAL_STEP / params.DATA_SAVE_STEP_PERIOD)),
        #     loss_ma_steps,
        #     label=label,
        #     linewidth=2.0 if n_step == 'Omega' else 1.0,
        #     linestyle='-' if n_step == 'Omega' else '--',
        #     color=colors[n_step_idx]
        # )
        # ax1.fill_between(
        #     range(int(params.MAX_GLOBAL_STEP / params.DATA_SAVE_STEP_PERIOD)),
        #     np.clip(loss_ma_steps - std_q_loss_ma_steps, 0, loss_ma_steps - std_q_loss_ma_steps),
        #     loss_ma_steps + std_q_loss_ma_steps,
        #     alpha=0.1,
        #     color=colors[n_step_idx]
        # )
        # ax1.tick_params(labelsize=24)

    ax0.set_xlabel('steps x ' + r'$10^3$', fontsize=32)
    ax0.set_ylabel('Episode Rewards', fontsize=32)     # Moving Average of Cumulative Rewards
    ax0.legend(loc="best", fontsize=32)
    ax0.grid()

    # ax1.set_xlabel('steps x ' + r'$10^3$', fontsize=24)
    # ax1.set_ylabel(r'TD'+'-Error', fontsize=24)
    # ax1.set_yscale('log')
    # ax1.legend(loc="best", fontsize=24)
    # ax1.grid()

    # plt.suptitle('Environment: {0} \n Multi-Step DQN'.format(params.ENVIRONMENT_ID.value), fontsize=24)
    plt.tight_layout()
    plt.gcf()
    plt.savefig("./Multi_Step_DQN_graph.png")
    plt.close()


def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    """
    if window >= len(values):
        if len(values) == 0:
            sma = 0.0
        else:
            sma = np.mean(np.asarray(values))
        a = [sma] * len(values)
    else:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
    return a


def env_episode_min_reward(env_name):
    if 'Pong' in env_name:
        return -21
    else:
        return 0


def save_data_as_pickle(episode_rewards_across_steps, q_loss_across_steps, env_name, EXPERIMENTS_N_STEPS, total_steps, episode):
    with open('{0}_{1}step.pickle'.format(env_name, EXPERIMENTS_N_STEPS), 'wb') as f:
        pickle.dump(
            {
                "episode_rewards_across_steps": episode_rewards_across_steps,
                "q_loss_across_steps": q_loss_across_steps,
            },
            f
        )
    with open('check_point.pickle', 'wb') as f:
        pickle.dump(
            {
                "total_steps": total_steps,
                "episode": episode
            },
            f
        )


def save_reward_as_pickle(episode_rewards_across_steps, params):
    with open('{0}_{1}-step_reward.pickle'.format(params.ENVIRONMENT_ID.value, 'Omega' if params.OMEGA else params.N_STEP), 'wb') as f:
        pickle.dump(
            {
                "episode_rewards_across_steps": episode_rewards_across_steps,
            },
            f
        )


def save_q_loss_as_pickle(q_loss_across_steps, params):
    with open('{0}_{1}-step_q_loss.pickle'.format(params.ENVIRONMENT_ID.value, 'Omega' if params.OMEGA else params.N_STEP), 'wb') as f:
        pickle.dump(
            {
                "q_loss_across_steps": q_loss_across_steps,
            },
            f
        )


def main():
    # with open('data.pickle', 'rb') as f:
    #     data = pickle.load(f)
    #
    # draw_graph(data['episode_rewards_across_steps'], data['q_loss_across_steps'], params)

    # episode_rewards_across_steps[n_steps][max_runs][max_steps / save_period]

    reward_data = np.zeros((len(EXPERIMENTS_N_STEPS), MAX_RUNS, int(params.MAX_GLOBAL_STEP / params.DATA_SAVE_STEP_PERIOD)))
    loss_data = np.zeros((len(EXPERIMENTS_N_STEPS), MAX_RUNS, int(params.MAX_GLOBAL_STEP / params.DATA_SAVE_STEP_PERIOD)))

    for i, n_step in enumerate(EXPERIMENTS_N_STEPS):
        for j in range(MAX_RUNS):
            with open('data/Freeway/{0}_{1}-step_reward_{2}.pickle'.format(params.ENVIRONMENT_ID.value, n_step, j + 1), 'rb') as f:
                data = pickle.load(f)
                reward_data[i][j] = data['episode_rewards_across_steps']

            with open('data/Freeway/{0}_{1}-step_q_loss_{2}.pickle'.format(params.ENVIRONMENT_ID.value, n_step, j + 1), 'rb') as f:
                data = pickle.load(f)
                loss_data[i][j] = data['q_loss_across_steps']

    draw_graph(reward_data, loss_data)


if __name__ == '__main__':
    main()
