from codes.d_agents.black_box.cma_es.cma_es_agent import AgentEMAES
from codes.f_main.general_main.a_common_main import *


def evaluate(env, model):
    observation = env.reset()
    episode_reward = 0.0
    steps = 0
    while True:
        observation_v = torch.FloatTensor([observation]).to(device)
        action_prob = model(observation_v)
        acts = action_prob.max(dim=1)[1]
        observation, reward, done, _ = env.step(acts.data.cpu().numpy()[0])
        episode_reward += reward
        steps += 1
        if done:
            break
    return episode_reward, steps


def evaluate_with_noise(env, model, noise):
    old_parameters = model.state_dict()
    for parameter, parameter_noise in zip(model.parameters(), noise):
        parameter.data += params.NOISE_STANDARD_DEVIATION * parameter_noise
    episode_reward, steps = evaluate(env, model)
    model.load_state_dict(old_parameters)
    return episode_reward, steps


def train_main():
    env = rl_utils.get_single_environment(params=params)
    input_shape, num_outputs, action_min, action_max = get_environment_input_output_info(env)

    agent = AgentEMAES(
        worker_id=-1, input_shape=input_shape, num_outputs=num_outputs,
        params=params, device=device
    )

    evaluation_idx = 0

    while True:
        batch_noises = []
        batch_episode_rewards = []
        batch_steps = 0
        for _ in range(params.MAX_BATCH_EPISODES):
            noises, neg_noises = agent.sample_noise()
            batch_noises.append(noises)
            batch_noises.append(neg_noises)

            episode_reward, steps = evaluate_with_noise(env, agent.model, noises)
            batch_episode_rewards.append(episode_reward)
            batch_steps += steps

            episode_reward, steps = evaluate_with_noise(env, agent.model, neg_noises)
            batch_episode_rewards.append(episode_reward)
            batch_steps += steps

            if batch_steps > params.MAX_BATCH_STEPS:
                break

        evaluation_idx += 1
        mean_episode_reward = np.mean(batch_episode_rewards)
        print("{0}: mean episode reward={1:.2f}".format(evaluation_idx, mean_episode_reward))

        if mean_episode_reward > 199:
            print("Solved in %d evaluations" % evaluation_idx)
            break
        else:
            agent.train_step(batch_noises, batch_episode_rewards)


if __name__ == "__main__":
    train_main()