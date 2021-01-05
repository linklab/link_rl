import datetime
import glob
import os, sys
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

GRAPH_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "graphs")
if not os.path.exists(GRAPH_SAVE_DIR):
    os.makedirs(GRAPH_SAVE_DIR)


def draw_performance(evaluate_steps, dqn_performance, random_performance):
    files = glob.glob(os.path.join(GRAPH_SAVE_DIR, "*"))
    for f in files:
        os.remove(f)

    plt.style.use('seaborn-dark-palette')

    plt.subplot(111)

    plt.plot(evaluate_steps, dqn_performance, label="Double+Dueling DQN")
    plt.plot(evaluate_steps, random_performance, label="Random")
    plt.ticklabel_format(useOffset=False)

    plt.ylabel("Profit")
    plt.xlabel("Step")
    plt.title("Profit Comparison")
    plt.legend(loc="best", fancybox=True, framealpha=0.3, fontsize=12)
    plt.grid(True)

    now = datetime.datetime.now()

    new_file_path = os.path.join(GRAPH_SAVE_DIR, "results_{0}.png".format(now.strftime("%Y_%m_%d_%H_%M")))
    plt.savefig(new_file_path)

    plt.clf()