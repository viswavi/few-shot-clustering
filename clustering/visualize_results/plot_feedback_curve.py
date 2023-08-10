'''
python plot_feedback_curve.py \
    --clustering-logs \
    ~/logs/clustering/pckmeans_newsgroups_sim3_50_constraints \
    ~/logs/clustering/pckmeans_newsgroups_sim3_100_constraints \
    ~/logs/clustering/pckmeans_newsgroups_sim3_200_constraints \
    ~/logs/clustering/pckmeans_newsgroups_sim3_500_constraints \
    --algorithms PCKMeans ActivePCKMeans \
    --metrics nmi general_pairwise_f1
'''

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--clustering-logs", action="extend", nargs="+", type=str)
parser.add_argument("--algorithms", action="extend", nargs="+", type=str)
parser.add_argument("--metrics", action="extend", nargs="+", type=str)

def plot_results(results_dict, metric, title, name_prefix, colors=['blue', 'green', 'red', 'yellow']):
    for i, algorithm in enumerate(results_dict):
        x_axis = []
        y_axis = []
        for num_constraints, metrics_dict in results_dict[algorithm].items():
            x_axis.append(num_constraints)
            y_axis.append(metrics_dict[metric])

        color = colors[i]
        accuracy_plot = plt.plot(x_axis, y_axis, 'o-r', color = color)
        accuracy_plot[0].set_label(algorithm)

    # max_line = plt.axhline(y = 0.81, color = 'r', linestyle = '--')
    # max_line.set_label("Optimal Clustering")
    plt.title(title)
    plt.xlabel("# of Pairwise Constraints")
    plt.ylabel(metric)
    plt.xlim(0, 1000)
    plt.ylim(0.1, 1.0)
    plt.legend()
    plt.savefig(f"{name_prefix}_{metric}.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()

    clustering_log_paths = args.clustering_logs
    num_constraints = []

    results_dict = {}
    for algorithm in args.algorithms:
        results_dict[algorithm] = {}
        for log_path in clustering_log_paths:
            num_constraints = int(log_path.split("_")[-2])
            if num_constraints not in results_dict[algorithm]:
                results_dict[algorithm][num_constraints] = {}

            log = open(log_path).readlines()
            results_dict_lines = []
            start_results = False
            for line in log:
                if line.strip() == "{":
                    start_results = True
                if start_results:
                    results_dict_lines.append(line)
            results_dict_json = "\n".join(results_dict_lines)
            logged_results = json.loads(results_dict_json)

            for metric in logged_results[algorithm]:
                if metric not in results_dict[algorithm][num_constraints]:
                    results_dict[algorithm][num_constraints][metric] = {}
                results_dict[algorithm][num_constraints][metric] = logged_results[algorithm][metric]["mean"]

    for metric in args.metrics:
        plot_results(results_dict,
                     metric=metric,
                     title="Our Clustering on Newsgroups-Sim-3",
                     name_prefix="/Users/vijay/Downloads/ours_newsgroups_sim3",
                     colors=['blue', 'green', 'red', 'yellow'])
        plot_results(results_dict,
                     metric=metric,
                     title="Our Clustering on Newsgroups-Sim-3",
                     name_prefix="/Users/vijay/Downloads/ours_newsgroups_sim3",
                     colors=['blue', 'green', 'red', 'yellow'])

    basu_et_al_results = {
        "ActivePCKMeans": {
            50: {
                "nmi": 0.17,
                "general_pairwise_f1": 0.44,
            },
            100: {
                "nmi": 0.328,
                "general_pairwise_f1": 0.495,
            },
            200: {
                "nmi": 0.47,
                "general_pairwise_f1": 0.61,
            },
            500: {
                "nmi": 0.56,
                "general_pairwise_f1": 0.68,
            },
            1000: {
                "nmi": 0.56,
                "general_pairwise_f1": 0.675,
            }
        },
        "PCKMeans": {
            50: {
                "nmi": 0.118,
                "general_pairwise_f1": 0.355,
            },
            100: {
                "nmi": 0.125,
                "general_pairwise_f1": 0.37,
            },
            200: {
                "nmi": 0.17,
                "general_pairwise_f1": 0.4,
            },
            500: {
                "nmi": 0.37,
                "general_pairwise_f1": 0.525,
            },
            1000: {
                "nmi": 0.488,
                "general_pairwise_f1": 0.64,
            }
        },
    }

    for metric in args.metrics:
        plot_results(basu_et_al_results,
                     metric=metric,
                     title="Basu et al 2014 on Newsgroups-Sim-3",
                     name_prefix="/Users/vijay/Downloads/basu_et_al_newsgroups_sim3",
                     colors=['orange', 'purple', 'teal', 'goldenrod'])
        plot_results(basu_et_al_results,
                     metric=metric,
                     title="Basu et al 2014 on Newsgroups-Sim-3",
                     name_prefix="/Users/vijay/Downloads/basu_et_al_newsgroups_sim3",
                     colors=['orange', 'purple', 'teal', 'goldenrod'])
