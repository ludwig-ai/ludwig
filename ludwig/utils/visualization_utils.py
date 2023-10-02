#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy
import logging
from collections import Counter
from sys import platform
from typing import Dict, List

import numpy as np
import pandas as pd
import ptitprince as pt
from packaging import version

from ludwig.constants import SPACE, TRAINING, VALIDATION

logger = logging.getLogger(__name__)

try:
    import matplotlib as mpl

    if platform == "darwin":  # OS X
        try:
            mpl.use("TkAgg")
        except ModuleNotFoundError:
            logger.warning("Unable to set TkAgg backend for matplotlib. Your Python may not be configured for Tk")
    import matplotlib.patches as patches
    import matplotlib.path as path
    import matplotlib.patheffects as PathEffects
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import ticker
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    raise RuntimeError(
        "matplotlib or seaborn are not installed. "
        "In order to install all visualization dependencies run "
        "pip install ludwig[viz]"
    ) from e

INT_QUANTILES = 10
FLOAT_QUANTILES = 10

# mapping from RayTune search space to Ludwig types (float, int, category) for hyperopt visualizations
RAY_TUNE_FLOAT_SPACES = {"uniform", "quniform", "loguniform", "qloguniform", "randn", "qrandn"}
RAY_TUNE_INT_SPACES = {"randint", "qrandint", "lograndint", "qlograndint"}
RAY_TUNE_CATEGORY_SPACES = {"choice", "grid_search"}

_matplotlib_34 = version.parse(mpl.__version__) >= version.parse("3.4")


def visualize_callbacks(callbacks, fig):
    if callbacks is None:
        return
    for callback in callbacks:
        callback.on_visualize_figure(fig)


def learning_curves_plot(
    train_values,
    vali_values,
    metric,
    x_label="epoch",
    x_step=1,
    algorithm_names=None,
    title=None,
    filename=None,
    callbacks=None,
):
    num_algorithms = len(train_values)
    max_len = max(len(tv) for tv in train_values)

    fig, ax = plt.subplots()

    sns.set_style("whitegrid")

    if title is not None:
        ax.set_title(title)

    if num_algorithms == 1:
        colors = plt.get_cmap("tab10").colors
    else:  # num_algorithms > 1
        colors = plt.get_cmap("tab20").colors

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace("_", " "))

    xs = np.arange(1, (max_len * x_step) + 1, x_step)

    for i in range(num_algorithms):
        name_prefix = algorithm_names[i] + " " if algorithm_names is not None and i < len(algorithm_names) else ""
        ax.plot(
            xs[: len(train_values[i])], train_values[i], label=name_prefix + TRAINING, color=colors[i * 2], linewidth=3
        )
        if i < len(vali_values) and vali_values[i] is not None and len(vali_values[i]) > 0:
            ax.plot(
                xs[: len(vali_values[i])],
                vali_values[i],
                label=name_prefix + VALIDATION,
                color=colors[i * 2 + 1],
                linewidth=3,
            )

    ax.legend()
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def compare_classifiers_plot(
    scores,
    metrics,
    algoritm_names=None,
    adaptive=False,
    decimals=4,
    title=None,
    filename=None,
    callbacks=None,
):
    assert len(scores) == len(metrics)
    assert len(scores) > 0

    num_metrics = len(metrics)

    sns.set_style("whitegrid")

    fig, ax = plt.subplots()

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)
    ax.set_xticklabels([], minor=True)

    if title is not None:
        ax.set_title(title)

    width = 0.8 / num_metrics if num_metrics > 1 else 0.4
    ticks = np.arange(len(scores[0]))

    if num_metrics <= 10:
        colors = plt.get_cmap("tab10").colors
    else:
        colors = plt.get_cmap("tab20").colors
    if adaptive:
        maximum = max(max(score) for score in scores)
    else:
        ax.set_xlim([0, 1])
        ax.set_xticks(np.linspace(0.0, 1.0, num=21), minor=True)
        ax.set_xticks(np.linspace(0.0, 1.0, num=11))
        maximum = 1

    half_total_width = 0.4 if num_metrics > 1 else 0.2
    ax.set_yticks(ticks + half_total_width - width / 2)
    ax.set_yticklabels(algoritm_names if algoritm_names is not None else "")
    ax.invert_yaxis()  # labels read top-to-bottom

    for i, metric in enumerate(metrics):
        ax.barh(ticks + (i * width), scores[i], width, label=metric, color=colors[i])

        for j, v in enumerate(scores[i]):
            if v < maximum * (0.025 * decimals + 0.1):
                x = v + maximum * 0.01
                horizontal_alignment = "left"
            else:
                x = v - maximum * 0.01
                horizontal_alignment = "right"
            txt = ax.text(
                x,
                ticks[j] + (i * width),
                ("{:." + str(decimals) + "f}").format(v),
                color="white",
                fontweight="bold",
                verticalalignment="center",
                horizontalalignment=horizontal_alignment,
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])

    plt.setp(ax.get_xminorticklabels(), visible=False)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def compare_classifiers_line_plot(
    xs,
    scores,
    metric,
    algorithm_names=None,
    title=None,
    filename=None,
    callbacks=None,
):
    assert len(scores) > 0

    sns.set_style("whitegrid")

    if len(scores) <= 10:
        colors = plt.get_cmap("tab10").colors
    else:
        colors = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots()

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)

    if title is not None:
        ax.set_title(title)

    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.set_xlabel("k")
    ax.set_ylabel(metric)

    for i, score in enumerate(scores):
        ax.plot(
            xs,
            score,
            label=algorithm_names[i] if algorithm_names is not None and i < len(algorithm_names) else f"Algorithm {i}",
            color=colors[i],
            linewidth=3,
            marker="o",
        )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def compare_classifiers_multiclass_multimetric_plot(
    scores,
    metrics,
    labels=None,
    title=None,
    filename=None,
    callbacks=None,
):
    assert len(scores) > 0

    sns.set_style("whitegrid")

    fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    width = 0.9 / len(scores)
    ticks = np.arange(len(scores[0]))

    if len(scores) <= 10:
        colors = plt.get_cmap("tab10").colors
    else:
        colors = plt.get_cmap("tab20").colors
    ax.set_xlabel("class")
    ax.set_xticks(ticks + width)
    if labels is not None:
        ax.set_xticklabels(labels, rotation=90)
    else:
        ax.set_xticklabels(ticks, rotation=90)

    for i, score in enumerate(scores):
        ax.bar(ticks + i * width, score, width, label=metrics[i], color=colors[i])

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def radar_chart(
    ground_truth,
    predictions,
    algorithms=None,
    log_scale=False,
    title=None,
    filename=None,
    callbacks=None,
):
    sns.set_style("whitegrid")

    if title is not None:
        plt.title(title)

    ground_truth = ground_truth[0:10]
    predictions = [pred[0:10] for pred in predictions]

    gt_argsort = np.argsort(-ground_truth)  # sort deacreasing
    logger.info(gt_argsort)
    ground_truth = ground_truth[gt_argsort]
    predictions = [pred[gt_argsort] for pred in predictions]

    maximum = max(max(ground_truth), max(max(p) for p in predictions))

    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rmax(maximum)
    ax.set_rlabel_position(305)
    ax.set_ylabel("Probability")
    # ax.set_rscale('log')
    ax.grid(True)

    colors = plt.get_cmap("tab10").colors

    num_classes = len(ground_truth)

    # Set ticks to the number of properties (in radians)
    t = np.arange(0, 2 * np.pi, 2 * np.pi / num_classes)
    if _matplotlib_34:
        ax.set_xticks(t)
    else:
        ax.set_xticks(t, [])
    ax.set_xticklabels(np.arange(0, num_classes))

    # Set yticks from 0 to 10
    # ax.set_yticks(np.linspace(0, 10, 11))
    # Set axes limits
    # ax.set_rlim(0, 1)
    # ax.set_rscale('log')

    def draw_polygon(values, label, color="grey"):
        points = [(x, y) for x, y in zip(t, values)]
        points.append(points[0])
        points = np.array(points)

        codes = [path.Path.MOVETO] + [path.Path.LINETO] * (len(values) - 1) + [path.Path.CLOSEPOLY]
        _path = path.Path(points, codes)
        _patch = patches.PathPatch(_path, fill=True, color=color, linewidth=0, alpha=0.2)
        ax.add_patch(_patch)
        _patch = patches.PathPatch(_path, fill=False, color=color, linewidth=3)
        ax.add_patch(_patch)

        # Draw circles at value points
        # line = ax.scatter(points[:, 0], points[:, 1], linewidth=3,
        #            s=50, color='white', edgecolor=color, zorder=10)
        ax.plot(
            points[:, 0],
            points[:, 1],
            linewidth=3,
            marker="o",
            fillstyle="full",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2,
            color=color,
            zorder=10,
            label=label,
        )

    draw_polygon(ground_truth, "Ground Truth")

    # Draw polygon representing values
    for i, alg_predictions in enumerate(predictions):
        draw_polygon(alg_predictions, algorithms[i], colors[i])

    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def pie(ax, values, **kwargs):
    total = sum(values)

    def formatter(pct):
        if pct > 0:
            return f"{pct * total / 100:0.0f}\n({pct:0.1f}%)"
        else:
            return ""

    wedges, _, labels = ax.pie(values, autopct=formatter, **kwargs)
    return wedges


def donut(
    inside_values,
    inside_labels,
    outside_values,
    outside_labels,
    outside_groups,
    title=None,
    tight_layout=None,
    filename=None,
    callbacks=None,
):
    fig, ax = plt.subplots(figsize=(7, 5))

    if title is not None:
        ax.set_title(title)

    ax.axis("equal")

    width = 0.35
    colors_tab20c = list(plt.get_cmap("tab20c").colors)
    colors_set2 = list(plt.get_cmap("Set2").colors)
    colors_set3 = list(plt.get_cmap("Set3").colors)
    colors_pastel1 = list(plt.get_cmap("Pastel1").colors)

    # swap green and red
    # for i in range(4):
    #    tmp = colors[4 + i]
    #    colors[4 + i] = colors[8 + i]
    #    colors[8 + i] = tmp

    colors = []
    colors.extend(colors_tab20c[8:12])
    colors.append(colors_set2[5])
    colors.append(colors_set3[11])
    colors.append(colors_set3[1])
    colors.append(colors_pastel1[5])
    colors.extend(colors_tab20c[4:8])

    inside_colors = [colors[x * 4] for x in range(len(inside_values))]

    group_count = Counter(outside_groups)
    outside_colors = [colors[(i * 4) + ((j % 3) + 1)] for i in list(set(outside_groups)) for j in range(group_count[i])]

    outside = pie(
        ax,
        outside_values,
        radius=1,
        pctdistance=1 - width / 2,
        colors=outside_colors,
        startangle=90,
        counterclock=False,
        textprops={
            "color": "w",
            "weight": "bold",
            "path_effects": [PathEffects.withStroke(linewidth=3, foreground="black")],
        },
    )
    inside = pie(
        ax,
        inside_values,
        radius=1 - width,
        pctdistance=1 - (width / 2) / (1 - width),
        colors=inside_colors,
        startangle=90,
        counterclock=False,
        textprops={
            "color": "w",
            "weight": "bold",
            "path_effects": [PathEffects.withStroke(linewidth=3, foreground="black")],
        },
    )
    plt.setp(inside + outside, width=width, edgecolor="white")

    wedges = []
    labels = []
    so_far = 0
    for i in list(set(outside_groups)):
        wedges.append(inside[i])
        labels.append(inside_labels[i])
        for j in range(group_count[i]):
            wedges.append(outside[so_far])
            labels.append(outside_labels[so_far])
            so_far += 1

    if tight_layout:
        ax.legend(wedges, labels, frameon=True, loc=1, bbox_to_anchor=(1.30, 1.00))
    else:
        ax.legend(wedges, labels, frameon=True, loc=1, bbox_to_anchor=(1.50, 1.00))
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()


def confidence_filtering_plot(
    thresholds,
    accuracies,
    dataset_kepts,
    algorithm_names=None,
    title=None,
    filename=None,
    callbacks=None,
):
    assert len(accuracies) == len(dataset_kepts)
    num_algorithms = len(accuracies)

    sns.set_style("whitegrid")

    if num_algorithms == 1:
        colors = plt.get_cmap("tab10").colors
    else:  # num_algorithms > 1
        colors = plt.get_cmap("tab20").colors

    y_ticks_minor = np.linspace(0.0, 1.0, num=21)
    y_ticks_major = np.linspace(0.0, 1.0, num=11)
    y_ticks_major_labels = [f"{y * 100:3.0f}%" for y in y_ticks_major]

    fig, ax1 = plt.subplots()

    if title is not None:
        ax1.set_title(title)

    ax1.grid(which="both")
    ax1.grid(which="minor", alpha=0.5)
    ax1.grid(which="major", alpha=0.75)
    ax1.set_xticks([x for idx, x in enumerate(thresholds) if idx % 2 == 0])
    ax1.set_xticks(thresholds, minor=True)

    ax1.set_xlim(-0.05, 1.05)
    ax1.set_xlabel("confidence threshold")

    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(y_ticks_major)
    ax1.set_yticklabels(y_ticks_major_labels)
    ax1.set_yticks(y_ticks_minor, minor=True)

    ax2 = ax1.twinx()

    ax2.set_ylim(0, 1.05)
    ax2.set_yticks(y_ticks_major)
    ax2.set_yticklabels(y_ticks_major_labels)
    ax2.set_yticks(y_ticks_minor, minor=True)

    for i in range(len(accuracies)):
        algorithm_name = algorithm_names[i] + " " if algorithm_names is not None and i < len(algorithm_names) else ""
        ax1.plot(thresholds, accuracies[i], label=f"{algorithm_name} accuracy", color=colors[i * 2], linewidth=3)
        ax1.plot(
            thresholds, dataset_kepts[i], label=f"{algorithm_name} data coverage", color=colors[i * 2 + 1], linewidth=3
        )

    ax1.legend(frameon=True, loc=3)
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def confidence_filtering_data_vs_acc_plot(
    accuracies,
    dataset_kepts,
    model_names=None,
    dotted=False,
    decimal_digits=0,
    y_label="accuracy",
    title=None,
    filename=None,
    callbacks=None,
):
    assert len(accuracies) == len(dataset_kepts)

    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors

    max_dataset_kept = max(max(dataset_kept) for dataset_kept in dataset_kepts)

    x_ticks_minor = np.linspace(0.0, max_dataset_kept, num=21)
    x_ticks_major = np.linspace(0.0, max_dataset_kept, num=11)
    x_ticks_major_labels = [
        "{value:3.{decimal_digits}f}%".format(decimal_digits=decimal_digits, value=x * 100) for x in x_ticks_major
    ]
    y_ticks_minor = np.linspace(0.0, 1.0, num=21)
    y_ticks_major = np.linspace(0.0, 1.0, num=11)

    fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)
    ax.set_xticks(x_ticks_major)
    ax.set_xticks(x_ticks_minor, minor=True)
    ax.set_xticklabels(x_ticks_major_labels)
    ax.set_xlim(0, max_dataset_kept)
    ax.set_xlabel("data coverage")

    ax.set_ylim(0, 1)
    ax.set_yticks(y_ticks_major)
    ax.set_yticks(y_ticks_minor, minor=True)
    ax.set_ylabel(y_label)

    for i in range(len(accuracies)):
        curr_dotted = dotted[i] if isinstance(dotted, (list, tuple)) and i < len(dotted) else dotted
        algorithm_name = model_names[i] + " " if model_names is not None and i < len(model_names) else ""
        ax.plot(
            dataset_kepts[i],
            accuracies[i],
            label=algorithm_name,
            color=colors[i],
            linewidth=3,
            linestyle=":" if curr_dotted else "-",
        )

    ax.legend(frameon=True, loc=3)
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def confidence_filtering_data_vs_acc_multiline_plot(
    accuracies,
    dataset_kepts,
    models_names,
    title=None,
    filename=None,
    callbacks=None,
):
    assert len(accuracies) == len(dataset_kepts)

    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab20").colors

    max_dataset_kept = max(max(dataset_kept) for dataset_kept in dataset_kepts)

    x_ticks_minor = np.linspace(0.0, max_dataset_kept, num=21)
    x_ticks_major = np.linspace(0.0, max_dataset_kept, num=11)
    x_ticks_major_labels = [f"{x * 100:3.0f}%" for x in x_ticks_major]
    y_ticks_minor = np.linspace(0.0, 1.0, num=21)
    y_ticks_major = np.linspace(0.0, 1.0, num=11)

    fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)
    ax.set_xticks(x_ticks_major)
    ax.set_xticks(x_ticks_minor, minor=True)
    ax.set_xticklabels(x_ticks_major_labels)
    ax.set_xlim(0, max_dataset_kept)
    ax.set_xlabel("data coverage")

    ax.set_ylim(0, 1)
    ax.set_yticks(y_ticks_major)
    ax.set_yticks(y_ticks_minor, minor=True)
    ax.set_ylabel("accuracy")

    for i in range(len(accuracies)):
        ax.plot(dataset_kepts[i], accuracies[i], color=colors[0], linewidth=1.0, alpha=0.35)

    legend_elements = [Line2D([0], [0], linewidth=1.0, color=colors[0])]
    ax.legend(legend_elements, models_names)
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def confidence_filtering_3d_plot(
    thresholds_1,
    thresholds_2,
    accuracies,
    dataset_kepts,
    threshold_output_feature_names=None,
    title=None,
    filename=None,
    callbacks=None,
):
    assert len(accuracies) == len(dataset_kepts)
    assert len(thresholds_1) == len(thresholds_2)

    thresholds_1, thresholds_2 = np.meshgrid(thresholds_1, thresholds_2)

    colors = plt.get_cmap("tab10").colors
    sns.set_style("white")

    z_ticks_minor = np.linspace(0.0, 1.0, num=21)
    z_ticks_major = np.linspace(0.0, 1.0, num=11)
    z_ticks_major_labels = [f"{z * 100:3.0f}%" for z in z_ticks_major]

    fig = plt.figure()
    ax = Axes3D
    ax = fig.add_subplot(111, projection="3d")

    if title is not None:
        ax.set_title(title)

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)

    ax.set_xlabel(f"{threshold_output_feature_names[0]} probability")
    ax.set_ylabel(f"{threshold_output_feature_names[1]} probability")

    ax.set_xlim(np.min(thresholds_1), np.max(thresholds_1))
    ax.set_ylim(np.min(thresholds_2), np.max(thresholds_2))
    ax.set_zlim(0, 1)
    ax.set_zticks(z_ticks_major)
    ax.set_zticklabels(z_ticks_major_labels)
    ax.set_zticks(z_ticks_minor, minor=True)

    # ORRIBLE HACK, IT'S THE ONLY WAY TO REMOVE PADDING
    from mpl_toolkits.mplot3d.axis3d import Axis

    if not hasattr(Axis, "_get_coord_info_old"):

        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs

        Axis._get_coord_info_old = Axis._get_coord_info
        Axis._get_coord_info = _get_coord_info_new
    # END OF HORRIBLE HACK

    surf_1 = ax.plot_surface(
        thresholds_1,
        thresholds_2,
        accuracies,
        alpha=0.5,
        label="accuracy",
        cmap=plt.get_cmap("winter"),
        edgecolor="none",
    )
    surf_2 = ax.plot_surface(
        thresholds_1,
        thresholds_2,
        dataset_kepts,
        alpha=0.5,
        label="data coverage",
        cmap=plt.get_cmap("autumn"),
        edgecolor="none",
    )

    handle_1 = copy.copy(surf_1)
    handle_2 = copy.copy(surf_2)

    handle_1.set_color(colors[0])
    handle_2.set_color(colors[1])

    # ## the next block is needed because matplotlib 3.3.3 renamed
    # _edgecolors3d -> _edgecolor3d
    # _facecolors3d -> _facecolor3d
    # but we want to try to keep compatibility with older versions
    # #### BEGIN COMPATIBILITY BLOCK #####
    if hasattr(handle_1, "_edgecolors3d"):
        edgecolor3d = handle_1._edgecolors3d
    else:
        edgecolor3d = handle_1._edgecolor3d
    handle_1._edgecolors2d = edgecolor3d
    handle_1._edgecolor2d = edgecolor3d

    if hasattr(handle_2, "_edgecolors3d"):
        edgecolor3d = handle_2._edgecolors3d
    else:
        edgecolor3d = handle_2._edgecolor3d
    handle_2._edgecolors2d = edgecolor3d
    handle_2._edgecolor2d = edgecolor3d

    if hasattr(handle_1, "_facecolors3d"):
        facecolor3d = handle_1._facecolors3d
    else:
        facecolor3d = handle_1._facecolor3d
    handle_1._facecolors2d = facecolor3d
    handle_1._facecolor2d = facecolor3d

    if hasattr(handle_2, "_facecolors3d"):
        facecolor3d = handle_2._facecolors3d
    else:
        facecolor3d = handle_2._facecolor3d
    handle_2._facecolors2d = facecolor3d
    handle_2._facecolor2d = facecolor3d
    # #### END COMPATIBILITY BLOCK #####

    ax.legend(frameon=True, loc=3, handles=[handle_1, handle_2])

    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def threshold_vs_metric_plot(
    thresholds,
    scores,
    algorithm_names=None,
    title=None,
    filename=None,
    callbacks=None,
):
    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors

    # y_ticks_minor = np.linspace(0.0, 1.0, num=21)
    # y_ticks_major = np.linspace(0.0, 1.0, num=11)
    # y_ticks_major_labels = ['{:3.0f}%'.format(y * 100) for y in y_ticks_major]

    fig, ax1 = plt.subplots()

    if title is not None:
        ax1.set_title(title)

    ax1.grid(which="both")
    ax1.grid(which="minor", alpha=0.5)
    ax1.grid(which="major", alpha=0.75)
    ax1.set_xticks([x for idx, x in enumerate(thresholds) if idx % 2 == 0])
    ax1.set_xticks(thresholds, minor=True)

    # ax1.set_xlim(0, 1)
    ax1.set_xlabel("confidence threshold")

    # ax1.set_ylim(0, 1)
    # ax1.set_yticks(y_ticks_major)
    # ax1.set_yticklabels(y_ticks_major_labels)
    # ax1.set_yticks(y_ticks_minor, minor=True)

    for i in range(len(scores)):
        algorithm_name = algorithm_names[i] + " " if algorithm_names is not None and i < len(algorithm_names) else ""
        ax1.plot(thresholds, scores[i], label=algorithm_name, color=colors[i], linewidth=3, marker="o")

    ax1.legend(frameon=True)
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def roc_curves(
    fpr_tprs,
    algorithm_names=None,
    title=None,
    graded_color=False,
    filename=None,
    callbacks=None,
):
    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors
    colormap = plt.get_cmap("RdYlGn")

    y_ticks_minor = np.linspace(0.0, 1.0, num=21)
    y_ticks_major = np.linspace(0.0, 1.0, num=11)

    fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)

    ax.set_xlim(0, 1)
    ax.set_xlabel("False positive rate")

    ax.set_ylim(0, 1)
    ax.set_yticks(y_ticks_major)
    ax.set_yticks(y_ticks_minor, minor=True)
    ax.set_ylabel("True positive rate")

    plt.plot([0, 1], [0, 1], color="black", linewidth=3, linestyle="--")

    for i in range(len(fpr_tprs)):
        algorithm_name = algorithm_names[i] + " " if algorithm_names is not None and i < len(algorithm_names) else ""
        color = colormap(i / len(fpr_tprs)) if graded_color else colors[i]
        ax.plot(fpr_tprs[i][0], fpr_tprs[i][1], label=algorithm_name, color=color, linewidth=3)

    ax.legend(frameon=True)
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def precision_recall_curves_plot(
    precision_recalls: Dict[str, List[float]],
    model_names: List[str],
    title: str = None,
    filename: str = None,
    callbacks=None,
):
    """Generates a precision recall curve for each model in the model_names list.

    Args:
        precision_recalls: A list of dictionaries representing the precision and recall values for each model
            in model_names. Each dictionary has two keys: "precisions" and "recalls".
    """
    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors

    _, ax = plt.subplots()

    ax.set_xlim(0, 1)
    # Create ticks for every 0.1 increment
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xlabel("Recall")

    ax.set_ylim(0, 1)
    # Create ticks for every 0.1 increment
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylabel("Precision")

    if title is not None:
        ax.set_title(title)

    for i in range(len(precision_recalls)):
        model_name = model_names[i] if model_names is not None and i < len(model_names) else ""
        ax.plot(
            precision_recalls[i]["recalls"],
            precision_recalls[i]["precisions"],
            label=model_name,
            color=colors[i],
            linewidth=3,
        )

    ax.legend(frameon=True)
    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def calibration_plot(
    fraction_positives,
    mean_predicted_values,
    algorithm_names=None,
    class_name=None,
    filename=None,
    callbacks=None,
):
    assert len(fraction_positives) == len(mean_predicted_values)

    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors

    num_algorithms = len(fraction_positives)

    plt.figure(figsize=(9, 9))
    plt.grid(which="both")
    plt.grid(which="minor", alpha=0.5)
    plt.grid(which="major", alpha=0.75)

    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for i in range(num_algorithms):
        # ax1.plot(mean_predicted_values[i], fraction_positives[i],
        #         label=algorithms[i] if algorithm_names is not None and i < len(algorithms) else '')

        # sns.tsplot(mean_predicted_values[i], fraction_positives[i], ax=ax1, color=colors[i])

        assert len(mean_predicted_values[i]) == len(fraction_positives[i])
        order = min(3, len(mean_predicted_values[i]) - 1)

        sns.regplot(
            x=mean_predicted_values[i],
            y=fraction_positives[i],
            order=order,
            x_estimator=np.mean,
            color=colors[i],
            marker="o",
            scatter_kws={"s": 40},
            label=algorithm_names[i] if algorithm_names is not None and i < len(algorithm_names) else f"Model {i}",
        )

    ticks = np.linspace(0.0, 1.0, num=11)
    plt.xlim([-0.05, 1.05])
    plt.xticks(ticks)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.ylim([-0.05, 1.05])
    plt.yticks(ticks)
    plt.legend(loc="lower right")
    if class_name is not None:
        plt.title(f"{class_name}: Calibration (reliability curve)")
    else:
        plt.title("Calibration (reliability curve)")

    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def brier_plot(
    brier_scores,
    algorithm_names=None,
    class_names=None,
    title=None,
    filename=None,
    callbacks=None,
):
    sns.set_style("whitegrid")

    # Dynamically set the size of the plot based on the number of labels
    # Use minimum size to prevent plot from being too small
    default_width, default_height = plt.rcParams.get("figure.figsize")
    width = max(default_width, len(class_names) / 2)
    height = max(default_height, len(class_names) / 2)
    fig, ax = plt.subplots(figsize=(width, height))

    if title is not None:
        plt.title(title)

    colors = plt.get_cmap("tab10").colors

    n_algorithms = brier_scores.shape[1]
    n_classes = brier_scores.shape[0]
    x = np.arange(n_classes)

    max_width = 0.35
    bar_width = min(0.5 / n_algorithms, max_width)
    bar_left = -bar_width * (n_algorithms // 2) + ((bar_width / 2) if (n_algorithms % 2) == 0 else 0)

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)
    ax.set_xlabel("class")
    ax.set_ylabel("brier score")
    if class_names is not None:
        ax.set_xticks(
            x,
            class_names,
            rotation=45,
            ha="center",
        )
    else:
        ax.set_xticks(
            x,
            [str(i) for i in range(n_classes)],
            rotation=45,
            ha="center",
        )

    for i in range(n_algorithms):
        # Plot bar for each class
        label = algorithm_names[i] if algorithm_names is not None and i < len(algorithm_names) else f"Model {i}"
        ax.bar(x + bar_left + (bar_width * i), brier_scores[:, i], bar_width, color=colors[i], label=label)

    ax.legend()
    fig.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def predictions_distribution_plot(
    probabilities,
    algorithm_names=None,
    filename=None,
    callbacks=None,
):
    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors

    num_algorithms = len(probabilities)

    plt.figure(figsize=(9, 9))
    plt.grid(which="both")
    plt.grid(which="minor", alpha=0.5)
    plt.grid(which="major", alpha=0.75)

    for i in range(num_algorithms):
        plt.hist(
            probabilities[i],
            range=(0, 1),
            bins=41,
            color=colors[i],
            label=algorithm_names[i] if algorithm_names is not None and i < len(algorithm_names) else "",
            histtype="stepfilled",
            alpha=0.5,
            lw=2,
        )

    plt.xlabel("Mean predicted value")
    plt.xlim([0, 1])
    plt.xticks(np.linspace(0.0, 1.0, num=21))
    plt.ylabel("Count")
    plt.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def confusion_matrix_plot(
    confusion_matrix,
    labels=None,
    output_feature_name=None,
    filename=None,
    callbacks=None,
):
    mpl.rcParams.update({"figure.autolayout": True})

    # Dynamically set the size of the plot based on the number of labels
    # Use minimum size to prevent plot from being too small
    default_width, default_height = plt.rcParams.get("figure.figsize")
    width = max(default_width, len(labels))
    height = max(default_height, len(labels))
    fig, ax = plt.subplots(figsize=(width, height))

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Set alpha value to prevent blue hues from being too dark
    cax = ax.matshow(confusion_matrix, cmap="Blues", alpha=0.6)
    # Annotate confusion matrix plot
    for (i, j), z in np.ndenumerate(confusion_matrix):
        # Format differently based on whether the value is normalized or not
        if z.is_integer():
            z_format = f"{z:.0f}"
        else:
            z_format = f"{z:.3f}"
        ax.text(
            j,
            i,
            z_format,
            ha="center",
            va="center",
            color="black",
            fontweight="medium",
            wrap=True,
        )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticklabels([""] + labels, rotation=45, ha="left")
    ax.set_yticklabels([""] + labels, rotation=45, ha="right")
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0)
    # https://stackoverflow.com/a/26720422/10102370 works nicely for square plots
    fig.colorbar(cax, ax=ax, extend="max", fraction=0.046, pad=0.04)
    ax.set_xlabel(f"Predicted {output_feature_name}")
    ax.set_ylabel(f"Actual {output_feature_name}")

    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()


def double_axis_line_plot(
    y1_sorted,
    y2,
    y1_name,
    y2_name,
    labels=None,
    title=None,
    filename=None,
    callbacks=None,
):
    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors

    # Dynamically adjust figure size based on number of labels
    default_width, default_height = plt.rcParams.get("figure.figsize")
    width = max(default_width, len(labels) / 3)
    height = max(default_height, len(labels) / 3)
    fig, ax1 = plt.subplots(layout="constrained", figsize=(width, height))

    if title is not None:
        ax1.set_title(title)

    ax1.set_xlabel(f"class (sorted by {y1_name})")
    ax1.set_xlim(0, len(y1_sorted) - 1)
    if labels is not None:
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_xticks(np.arange(len(labels)))

    ax1.set_ylabel(y1_name, color=colors[1])
    ax1.tick_params("y", colors=colors[1])
    ax1.set_ylim(min(y1_sorted), max(y1_sorted))

    ax2 = ax1.twinx()
    ax2.set_ylabel(y2_name, color=colors[0])
    ax2.tick_params("y", colors=colors[0])
    ax2.set_ylim(min(y2), max(y2))

    ax1.plot(y1_sorted, label=y1_name, color=colors[1], linewidth=4)
    ax2.plot(y2, label=y2_name, color=colors[0], linewidth=3)

    fig.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_matrix(
    matrix,
    cmap="hot",
    filename=None,
    callbacks=None,
):
    plt.figure()
    plt.matshow(matrix, cmap=cmap)
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_distributions(
    distributions,
    labels=None,
    title=None,
    filename=None,
    callbacks=None,
):
    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors

    fig, ax1 = plt.subplots()

    if title is not None:
        ax1.set_title(title)

    ax1.grid(which="both")
    ax1.grid(which="minor", alpha=0.5)
    ax1.grid(which="major", alpha=0.75)

    ax1.set_xlabel("class")

    ax1.set_ylabel("p")
    ax1.tick_params("y")

    for i, distribution in enumerate(distributions):
        ax1.plot(
            distribution,
            color=colors[i],
            alpha=0.6,
            label=labels[i] if labels is not None and i < len(labels) else f"Distribution {i}",
        )

    ax1.legend(frameon=True)
    fig.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def plot_distributions_difference(
    distribution,
    labels=None,
    title=None,
    filename=None,
    callbacks=None,
):
    sns.set_style("whitegrid")

    colors = plt.get_cmap("tab10").colors

    fig, ax1 = plt.subplots()

    if title is not None:
        ax1.set_title(title)

    ax1.grid(which="both")
    ax1.grid(which="minor", alpha=0.5)
    ax1.grid(which="major", alpha=0.75)

    ax1.set_xlabel("class")

    ax1.set_ylabel("p")
    ax1.tick_params("y")

    ax1.plot(distribution, color=colors[0])

    fig.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def bar_plot(
    xs,
    ys,
    decimals=4,
    labels=None,
    title=None,
    filename=None,
    callbacks=None,
):
    assert len(xs) == len(ys)
    assert len(xs) > 0

    sns.set_style("whitegrid")

    # Dynamically set the size of the plot based on the number of labels
    # Use minimum size to prevent plot from being too small
    default_width, default_height = plt.rcParams.get("figure.figsize")
    width = max(default_width, len(labels) / 2)
    _, ax = plt.subplots(figsize=(width, default_height))

    ax.grid(which="both")
    ax.grid(which="minor", alpha=0.5)
    ax.grid(which="major", alpha=0.75)

    if title is not None:
        ax.set_title(title)

    colors = plt.get_cmap("tab10").colors

    ax.invert_yaxis()  # labels read top-to-bottom

    maximum = ys.max()
    ticks = np.arange(len(xs))
    ax.set_yticks(ticks)
    if labels is None:
        ax.set_yticklabels(xs)
    else:
        ax.set_yticklabels(labels)

    ax.barh(ticks, ys, color=colors[0], align="center")

    for i, v in enumerate(ys):
        if v < maximum * (0.025 * decimals + 0.1):
            x = v + maximum * 0.01
            horizontal_alignment = "left"
        else:
            x = v - maximum * 0.01
            horizontal_alignment = "right"
        txt = ax.text(
            x,
            ticks[i],
            ("{:." + str(decimals) + "f}").format(v),
            color="white",
            fontweight="bold",
            verticalalignment="center",
            horizontalalignment=horizontal_alignment,
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="black")])

    plt.tight_layout()
    visualize_callbacks(callbacks, plt.gcf())
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def hyperopt_report(hyperparameters, hyperopt_results_df, metric, filename_template, float_precision=3):
    title = "Hyperopt Report: {}"
    for hp_name, hp_params in hyperparameters.items():
        if hp_params[SPACE] in RAY_TUNE_INT_SPACES:
            hyperopt_int_plot(
                hyperopt_results_df,
                hp_name,
                metric,
                title.format(hp_name),
                filename_template.format(hp_name) if filename_template else None,
            )
        elif hp_params[SPACE] in RAY_TUNE_FLOAT_SPACES:
            hyperopt_float_plot(
                hyperopt_results_df,
                hp_name,
                metric,
                title.format(hp_name),
                filename_template.format(hp_name) if filename_template else None,
                log_scale_x=hp_params["scale"] == "log" if "scale" in hp_params else False,
            )
        elif hp_params[SPACE] in RAY_TUNE_CATEGORY_SPACES:
            hyperopt_category_plot(
                hyperopt_results_df,
                hp_name,
                metric,
                title.format(hp_name),
                filename_template.format(hp_name) if filename_template else None,
            )
        else:
            # TODO: more research needed on how to handle RayTune "sample_from" search space
            raise ValueError(
                f"{hp_params[SPACE]} search space not supported in Ludwig.  "
                f"Supported values are {RAY_TUNE_FLOAT_SPACES | RAY_TUNE_INT_SPACES | RAY_TUNE_CATEGORY_SPACES}."
            )

    # quantize float and int columns
    for hp_name, hp_params in hyperparameters.items():
        if hp_params[SPACE] in RAY_TUNE_INT_SPACES:
            num_distinct_values = len(hyperopt_results_df[hp_name].unique())
            if num_distinct_values > INT_QUANTILES:
                hyperopt_results_df[hp_name] = pd.qcut(hyperopt_results_df[hp_name], q=INT_QUANTILES, precision=0)
        elif hp_params[SPACE] in RAY_TUNE_FLOAT_SPACES:
            hyperopt_results_df[hp_name] = pd.qcut(
                hyperopt_results_df[hp_name],
                q=FLOAT_QUANTILES,
                precision=float_precision,
                duplicates="drop",
            )

    hyperopt_pair_plot(
        hyperopt_results_df,
        metric,
        title.format("pair plot"),
        filename_template.format("pair_plot") if filename_template else None,
    )


def hyperopt_int_plot(hyperopt_results_df, hp_name, metric, title, filename, log_scale_x=False, log_scale_y=True):
    sns.set_style("whitegrid")
    plt.figure()
    seaborn_figure = sns.scatterplot(x=hp_name, y=metric, data=hyperopt_results_df)
    seaborn_figure.set_title(title)
    if log_scale_x:
        seaborn_figure.set(xscale="log")
    if log_scale_y:
        seaborn_figure.set(yscale="log")
    seaborn_figure.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    seaborn_figure.xaxis.set_major_formatter(ticker.ScalarFormatter())
    seaborn_figure.xaxis.set_minor_formatter(ticker.NullFormatter())
    seaborn_figure.figure.tight_layout()
    if filename:
        seaborn_figure.figure.savefig(filename)
    else:
        seaborn_figure.figure.show()


def hyperopt_float_plot(hyperopt_results_df, hp_name, metric, title, filename, log_scale_x=False, log_scale_y=True):
    sns.set_style("whitegrid")
    plt.figure()
    seaborn_figure = sns.scatterplot(x=hp_name, y=metric, data=hyperopt_results_df)
    seaborn_figure.set_title(title)
    seaborn_figure.set(ylabel=metric)
    if log_scale_x:
        seaborn_figure.set(xscale="log")
    if log_scale_y:
        seaborn_figure.set(yscale="log")
    seaborn_figure.figure.tight_layout()
    if filename:
        seaborn_figure.figure.savefig(filename)
    else:
        seaborn_figure.figure.show()


def hyperopt_category_plot(hyperopt_results_df, hp_name, metric, title, filename, log_scale=True):
    sns.set_style("whitegrid")
    plt.figure()

    # Ensure that all parameter values have at least 2 trials, otherwise the Raincloud Plot will create awkward
    # looking "flat clouds" in the cloud part of the plot (the "rain" part is ok with 1 trial). In this case,
    # just use stripplots since they are categorical scatter plots.
    parameter_to_trial_count = hyperopt_results_df[hp_name].value_counts()
    parameter_to_trial_count = parameter_to_trial_count[parameter_to_trial_count < 2]

    if len(parameter_to_trial_count) != 0:
        seaborn_figure = sns.stripplot(x=hp_name, y=metric, data=hyperopt_results_df, size=5)
    else:
        seaborn_figure = pt.RainCloud(
            x=hp_name,
            y=metric,
            data=hyperopt_results_df,
            palette="Set2",
            bw=0.2,
            width_viol=0.7,
            point_size=6,
            cut=1,
        )

    seaborn_figure.set_title(title)
    seaborn_figure.set(ylabel=metric)
    sns.despine()
    if log_scale:
        seaborn_figure.set(yscale="log")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def hyperopt_pair_plot(hyperopt_results_df, metric, title, filename):
    params = sorted(list(hyperopt_results_df.keys()))
    params.remove(metric)
    num_param = len(params)

    # Pair plot is empty if there's only 1 parameter, so skip creating a pair plot
    if num_param == 1:
        return

    sns.set_style("white")
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(title)
    gs = fig.add_gridspec(num_param, num_param)

    for i, param1 in enumerate(params):
        for j, param2 in enumerate(params):
            if i != j:
                ax = fig.add_subplot(gs[i, j])
                heatmap = hyperopt_results_df.pivot_table(index=param1, columns=param2, values=metric, aggfunc="mean")
                sns.heatmap(
                    heatmap,
                    linewidths=1,
                    cmap="viridis",
                    cbar_kws={"label": metric},
                    ax=ax,
                )

    plt.tight_layout(pad=5)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def hyperopt_hiplot(
    hyperopt_df,
    filename,
):
    import hiplot as hip

    experiment = hip.Experiment.from_dataframe(hyperopt_df)
    experiment.to_html(filename)
