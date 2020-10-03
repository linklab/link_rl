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
        self.loss_entropy = self.vis.line(X=[0], Y=[0], opts=dict(title="loss entropy"))
        self.loss_policy = self.vis.line(X=[0], Y=[0], opts=dict(title="loss policy"))
        self.loss_total = self.vis.line(X=[0], Y=[0], opts=dict(title="loss total"))
        self.grad_means = self.vis.line(X=[0], Y=[0], opts=dict(title="gradient means"))
        self.grad_max = self.vis.line(X=[0], Y=[0], opts=dict(title="gradient max"))

    def draw_optimization_performance(self, global_step, kl_divergence, baseline, mean_batch_scale,
                                      entropy, loss_entropy, loss_policy, loss_total, grad_means, grad_max):
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
            X=[global_step], Y=[grad_means], win=self.grad_means, name="gradient means", update="append",
            opts=dict(title='gradient means', showlegend=False)
        )

        self.vis.line(
            X=[global_step], Y=[grad_max], win=self.grad_max, name="gradient max", update="append",
            opts=dict(title='gradient max', showlegend=False)
        )