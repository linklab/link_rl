from visdom import Visdom

# python -m visdom.server
# http://localhost:8097
class Statistics:
    def __init__(self, method, args):
        self.args = args
        self.method = method

        self.vis_dom_info_text = "<div style='margin:1.0em;font-size:1.5em'>Env: {0}<br/>Method: {1}<br/> Global Step: {2}</div>"
        self.vis = None
        self.plt = None

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


class StatisticsForModelLoss:
    def __init__(self, args):
        self.args = args
        self.plt = None

        self.vis = Visdom()

        self.plt = self.vis.line(X=[0], Y=[0], opts=dict(title="model loss"))

    def draw_loss(self, global_step, model_loss):
        self.vis.line(
            X=[global_step], Y=[model_loss], win=self.plt, name="model_loss", update="append",
            opts=dict(title='model loss', showlegend=False)
        )