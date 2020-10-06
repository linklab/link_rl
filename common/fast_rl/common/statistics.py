from visdom import Visdom

# python -m visdom.server
# http://localhost:8097
class StatisticsForValueBasedRL:
    def __init__(self, method):
        self.method = method

        self.vis_dom_info_text = "<div style='margin:1.0em;font-size:1.5em'>Env: {0}<br/>Method: {1}<br/> Global Step: {2}</div>"

        self.vis = Visdom()
        self.vis.delete_env("main")

        self.plt_titles = [
            "mean episode reward", "speed", "epsilon"
        ]
        self.plt_names = [
            "mean episode reward", "speed", "epsilon"
        ]
        self.plt_opts = [
            dict(title='mean episode reward', showlegend=False),
            dict(title='speed', showlegend=False),
            dict(title='epsilon', showlegend=False)
        ]

        self.plt = {}

        for i in range(3):
            self.plt[i] = self.vis.line(X=[0], Y=[0], opts=dict(title=self.plt_titles[i]))

    def draw_performance(self, global_step, mean_episode_reward, speed, epsilon):
        self.vis.line(
            X=[global_step], Y=[mean_episode_reward], win=self.plt[0], name=self.plt_names[0], update="append",
            opts=self.plt_opts[0]
        )

        self.vis.line(
            X=[global_step], Y=[speed], win=self.plt[1], name=self.plt_names[1], update="append",
            opts=self.plt_opts[1]
        )

        self.vis.line(
            X=[global_step], Y=[epsilon], win=self.plt[2], name=self.plt_names[2], update="append",
            opts=self.plt_opts[2]
        )

    def conditional_save_model(self, current_global_step, q_nets, model_save_path, periodic_model_save_path, epsilon, method):
        pass


class StatisticsForPolicyBasedRL:
    def __init__(self, method):
        self.vis = Visdom()

        self.method = method
        self.vis_dom_info_text = "<div style='margin:1.0em;font-size:1.5em'>Env: {0}<br/>Method: {1}<br/> Global Step: {2}</div>"

        self.mean_episode_reward = self.vis.line(X=[0], Y=[0], opts=dict(title="mean episode reward"))
        self.speed = self.vis.line(X=[0], Y=[0], opts=dict(title="speed"))

    def draw_performance(self, global_step, mean_episode_reward, speed):
        self.vis.line(
            X=[global_step], Y=[mean_episode_reward], win=self.mean_episode_reward, name="mean_episode_reward", update="append",
            opts=dict(title='mean_episode_reward', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[speed], win=self.speed, name="speed", update="append",
            opts=dict(title='speed', showlegend=False)
        )

    def conditional_save_model(self, current_global_step, q_nets, model_save_path, periodic_model_save_path, epsilon, method):
        pass

########################################################################################################################
########################################################################################################################

class StatisticsForValueBasedOptimization:
    def __init__(self):
        self.vis = Visdom()

        self.plt = self.vis.line(X=[0], Y=[0], opts=dict(title="model loss"))

    def draw_optimization_performance(self, global_step, model_loss):
        self.vis.line(
            X=[global_step], Y=[model_loss], win=self.plt, name="model_loss", update="append",
            opts=dict(title='model loss', showlegend=False)
        )


class StatisticsForPolicyBasedRLOptimization:
    def __init__(self):
        self.vis = Visdom()

        self.kl_divergence = self.vis.line(X=[0], Y=[0], opts=dict(title="kl divergence"))
        self.baseline = self.vis.line(X=[0], Y=[0], opts=dict(title="baseline"))
        self.mean_batch_scale = self.vis.line(X=[0], Y=[0], opts=dict(title="mean batch scale"))
        self.entropy = self.vis.line(X=[0], Y=[0], opts=dict(title="entropy"))
        self.loss_policy = self.vis.line(X=[0], Y=[0], opts=dict(title="loss policy"))
        self.loss_entropy = self.vis.line(X=[0], Y=[0], opts=dict(title="loss entropy"))
        self.loss_total = self.vis.line(X=[0], Y=[0], opts=dict(title="loss total"))
        self.grad_l2 = self.vis.line(X=[0], Y=[0], opts=dict(title="gradient l2"))
        self.grad_variance = self.vis.line(X=[0], Y=[0], opts=dict(title="gradient variance"))
        self.grad_max = self.vis.line(X=[0], Y=[0], opts=dict(title="gradient max"))

    def draw_optimization_performance(self, global_step, kl_divergence, baseline, mean_batch_scale,
                                      entropy, loss_policy, loss_entropy, loss_total, grad_l2, grad_variance, grad_max):
        self.vis.line(
            X=[global_step], Y=[kl_divergence], win=self.kl_divergence, name="kl divergence", update="append",
            opts=dict(title='kl divergence', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[baseline], win=self.baseline, name="baseline", update="append",
            opts=dict(title='baseline', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[mean_batch_scale], win=self.mean_batch_scale, name="mean batch scale", update="append",
            opts=dict(title='mean batch scale', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[entropy], win=self.entropy, name="entropy", update="append",
            opts=dict(title='entropy', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_entropy], win=self.loss_entropy, name="loss entropy", update="append",
            opts=dict(title='loss entropy', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_policy], win=self.loss_policy, name="loss policy", update="append",
            opts=dict(title='loss policy', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_total], win=self.loss_total, name="loss total", update="append",
            opts=dict(title='loss total', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[grad_l2], win=self.grad_l2, name="gradient l2", update="append",
            opts=dict(title='gradient l2', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[grad_variance], win=self.grad_variance, name="gradient variance", update="append",
            opts=dict(title='gradient variance', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[grad_max], win=self.grad_max, name="gradient max", update="append",
            opts=dict(title='gradient max', showlegend=False)
        )


class StatisticsForA2COptimization:
    def __init__(self):
        self.vis = Visdom()

        self.kl_divergence = self.vis.line(X=[0], Y=[0], opts=dict(title="kl divergence"))
        self.mean_advantage = self.vis.line(X=[0], Y=[0], opts=dict(title="mean advantage"))
        self.entropy = self.vis.line(X=[0], Y=[0], opts=dict(title="entropy"))
        self.loss_actor = self.vis.line(X=[0], Y=[0], opts=dict(title="loss actor"))
        self.loss_critic = self.vis.line(X=[0], Y=[0], opts=dict(title="loss critic"))
        self.loss_entropy = self.vis.line(X=[0], Y=[0], opts=dict(title="loss entropy"))
        self.loss_total = self.vis.line(X=[0], Y=[0], opts=dict(title="loss total"))
        self.grad_l2 = self.vis.line(X=[0], Y=[0], opts=dict(title="gradient l2"))
        self.grad_variance = self.vis.line(X=[0], Y=[0], opts=dict(title="gradient variance"))
        self.grad_max = self.vis.line(X=[0], Y=[0], opts=dict(title="gradient max"))

    def draw_optimization_performance(self, global_step, kl_divergence, mean_advantage,
                                      entropy, loss_actor, loss_critic, loss_entropy, loss_total, grad_l2, grad_variance, grad_max):
        self.vis.line(
            X=[global_step], Y=[kl_divergence], win=self.kl_divergence, name="kl divergence", update="append",
            opts=dict(title='kl divergence', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[mean_advantage], win=self.mean_advantage, name="mean advantage", update="append",
            opts=dict(title='mean advantage', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[entropy], win=self.entropy, name="entropy", update="append",
            opts=dict(title='entropy', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_actor], win=self.loss_actor, name="loss actor", update="append",
            opts=dict(title='loss actor', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_critic], win=self.loss_critic, name="loss critic", update="append",
            opts=dict(title='loss critic', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_entropy], win=self.loss_entropy, name="loss entropy", update="append",
            opts=dict(title='loss entropy', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_total], win=self.loss_total, name="loss total", update="append",
            opts=dict(title='loss total', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[grad_l2], win=self.grad_l2, name="gradient l2", update="append",
            opts=dict(title='gradient l2', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[grad_variance], win=self.grad_variance, name="gradient variance", update="append",
            opts=dict(title='gradient variance', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[grad_max], win=self.grad_max, name="gradient max", update="append",
            opts=dict(title='gradient max', showlegend=False)
        )


class StatisticsForDDPGOptimization:
    def __init__(self, n_actions):
        self.vis = Visdom()

        self.loss_actor = self.vis.line(X=[0], Y=[0], opts=dict(title="loss actor"))
        self.loss_critic = self.vis.line(X=[0], Y=[0], opts=dict(title="loss critic"))
        self.loss_total = self.vis.line(X=[0], Y=[0], opts=dict(title="loss total"))
        self.actor_grad_l2 = self.vis.line(X=[0], Y=[0], opts=dict(title="actor gradient l2"))
        self.actor_grad_variance = self.vis.line(X=[0], Y=[0], opts=dict(title="actor gradient variance"))
        self.actor_grad_max = self.vis.line(X=[0], Y=[0], opts=dict(title="actor gradient max"))
        self.critic_grad_l2 = self.vis.line(X=[0], Y=[0], opts=dict(title="critic gradient l2"))
        self.critic_grad_variance = self.vis.line(X=[0], Y=[0], opts=dict(title="critic gradient variance"))
        self.critic_grad_max = self.vis.line(X=[0], Y=[0], opts=dict(title="critic gradient max"))
        self.buffer_length = self.vis.line(X=[0], Y=[0], opts=dict(title="buffer length"))
        self.n_actions = n_actions
        self.actions = {}
        for i in range(n_actions):
            self.actions[i] = self.vis.line(X=[0], Y=[0], opts=dict(title="action {0}".format(i)))

    def draw_optimization_performance(self, global_step, loss_actor, loss_critic, loss_total,
                                      actor_grad_l2, actor_grad_variance, actor_grad_max,
                                      critic_grad_l2, critic_grad_variance, critic_grad_max, buffer_length, actions):
        self.vis.line(
            X=[global_step], Y=[loss_actor], win=self.loss_actor, name="loss actor", update="append",
            opts=dict(title='loss actor', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_critic], win=self.loss_critic, name="loss critic", update="append",
            opts=dict(title='loss critic', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[loss_total], win=self.loss_total, name="loss total", update="append",
            opts=dict(title='loss total', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[actor_grad_l2], win=self.actor_grad_l2, name="actor gradient l2", update="append",
            opts=dict(title='actor gradient l2', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[actor_grad_variance], win=self.actor_grad_variance, name="actor gradient variance", update="append",
            opts=dict(title='actor gradient variance', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[actor_grad_max], win=self.actor_grad_max, name="actor gradient max", update="append",
            opts=dict(title='actor gradient max', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[critic_grad_l2], win=self.critic_grad_l2, name="critic gradient l2", update="append",
            opts=dict(title='critic gradient l2', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[critic_grad_variance], win=self.critic_grad_variance, name="critic gradient variance", update="append",
            opts=dict(title='critic gradient variance', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[critic_grad_max], win=self.critic_grad_max, name="critic gradient max", update="append",
            opts=dict(title='critic gradient max', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[buffer_length], win=self.buffer_length, name="buffer length", update="append",
            opts=dict(title='buffer length', showlegend=False)
        )

        for i in range(self.n_actions):
            self.vis.line(
                X=[global_step], Y=[actions[i]], win=self.actions[i], name="action {0}".format(i), update="append",
                opts=dict(title="action {0}".format(i), showlegend=False)
            )