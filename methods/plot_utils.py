import pickle
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy import stats

from methods.func_utils import test_function

COLORS = ["#0072B2", "#E69F00", "#009E73", "3", "4", "#D55E00", "6", "#7f7f7f"]


def plot_raw_function(x_train, y_train, x_test, y_test, y_results=None, loss=None):
    """Plots truncated Fourier series."""
    if loss is None:
        fig = plt.figure(figsize=(10, 5))
        ax = [plt.subplot2grid((1, 1), (0, 0))]
    else:
        fig = plt.figure(figsize=(15, 5))
        ax = [plt.subplot2grid((1, 2), (0, 0)), plt.subplot2grid((1, 2), (0, 1))]
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("y_loss")
        ax[1].plot(loss, c="red", label="loss")
        ax[1].legend()

    if y_results is None:
        ax[0].scatter(
            x_test, y_train, facecolor="white", edgecolor=COLORS[0], label="train"
        )
        ax[0].scatter(
            x_test, y_test, facecolor="white", edgecolor="black", label="test"
        )
        ax[0].set_ylim(-1, 1)
    else:
        ax[0].plot(x_test, y_results, c=COLORS[0], label="model")
        ax[0].scatter(
            x_test, y_test, facecolor="white", edgecolor="black", label="test"
        )
        ax[0].set_ylim(-1, 1)

    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend()


def get_experiments_and_configs(filename):
    """Returns ansaetze, configs and coeffs_file
    from stored trained function file."""
    with open(filename, "rb") as f:
        d = pickle.load(f)

    return d.get("ansaetze", None), d.get("configs", None), d.get("coeffs_file", None)


def get_function_values(filename):
    """Returns x_test, y_test and coeffs from stored trained function file."""
    with open(filename, "rb") as f:
        d = pickle.load(f)

    return d.get("x_test", None), d.get("y_test", None), d.get("coeffs", None)


def get_model_results(filename):
    """Returns model_results, model_parameters and model_histories
    from stored trained function file."""
    with open(filename, "rb") as f:
        d = pickle.load(f)

    return (
        d.get("model_results", None),
        d.get("model_parameters", None),
        d.get("model_histories", None),
    )


def get_name(ansatz, parameter=None):
    name = ""

    for i, feature in enumerate(ansatz):
        if i != 0:
            name += ","
        name += str(feature)

    if parameter is not None:
        name += "," + str(parameter)

    return name


def get_data(functs_files):
    """Returns all necessary data from stored trained function file."""
    run_names = list(functs_files.keys())
    file_names = list(functs_files.values())

    all_ansaetze = []
    all_y_test = []
    coeffs_files_list = []

    all_model_results = []
    all_model_parameters = []
    all_model_histories = []

    all_configs = []

    for i, file_name in enumerate(file_names):
        file_name_path = file_name + ".pickle"

        ansaetze, configs, coeffs_f = get_experiments_and_configs(file_name_path)
        x_test, y_test, _ = get_function_values(file_name_path)
        model_results, model_parameters, model_histories = get_model_results(
            file_name_path
        )

        all_ansaetze.append(ansaetze)
        coeffs_files_list.append(coeffs_f)
        all_y_test.append(y_test)

        all_configs.append(configs)
        all_model_results.append(model_results)
        all_model_parameters.append(model_parameters)
        all_model_histories.append(model_histories)

    all_model_histories_length = np.full(
        (
            np.shape(all_model_histories)[0],
            np.shape(all_model_histories)[1],
            np.shape(all_model_histories)[2],
        ),
        1,
    )

    all_model_histories_last = np.full(
        (
            np.shape(all_model_histories)[0],
            np.shape(all_model_histories)[1],
            np.shape(all_model_histories)[2],
        ),
        1.0,
    )

    for nb_data in range(np.shape(all_model_histories)[0]):
        for nb_config in range(np.shape(all_model_histories)[1]):
            for nb_func in range(np.shape(all_model_histories)[2]):
                all_model_histories_length[nb_data][nb_config][nb_func] = len(
                    all_model_histories[nb_data][nb_config][nb_func]
                )
                all_model_histories_last[nb_data][nb_config][
                    nb_func
                ] = all_model_histories[nb_data][nb_config][nb_func][-1]

    return (
        coeffs_files_list,
        all_model_histories_length,
        all_model_histories_last,
        all_model_parameters,
        all_ansaetze,
        run_names,
        file_names,
        x_test,
        all_y_test,
        all_model_results,
    )


def get_conf(values):
    """Calculates 0.95 confidenz intervall based on t-distribution."""
    n = len(values)
    sem = stats.sem(values)
    conf = sem * stats.t.ppf(1.95 / 2.0, n - 1)
    return conf


def find_nearest(array, value):
    """Finds vlaue in array that is closest to the average value."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_lc_with_avg(
    x_test,
    all_y_test,
    all_model_results,
    all_model_histories_last,
    filename="learning_capabiltiy",
    save_path="results/own/img/",
):
    """Plots learning capability."""
    fig_cols = 2
    fig_rows = 1
    fig = plt.figure(figsize=(5, 2))

    ax = []
    for r in range(fig_rows):
        ax_row = []
        for c in range(fig_cols):
            ax_row.append(plt.subplot2grid((fig_rows, fig_cols), (r, c)))

        ax.append(ax_row)

    name = "config"
    values = all_model_histories_last[0]
    mean = np.mean(values, axis=1)
    conf = get_conf(values[0])

    ax[0][0].set_title("learning capability = " + "{:0.3e}".format(mean[0]))
    ax[0][0].errorbar(
        name, mean[0], conf, marker="o", ecolor=COLORS[0], capsize=10, color=COLORS[0]
    )
    y_locator_factor = 1
    y_locator = [
        y_locator_factor * 1e-1,
        y_locator_factor * 1e-2,
        y_locator_factor * 1e-3,
        y_locator_factor * 1e-4,
        y_locator_factor * 1e-5,
    ]
    ax[0][0].set_yscale("log")
    ax[0][0].set_ylim(1e-5, 1e-1)
    ax[0][0].yaxis.set_major_locator(FixedLocator(y_locator))
    ax[0][0].tick_params(axis="both", direction="in")

    arg_avg = find_nearest(all_model_histories_last[0], mean)
    y_test = all_y_test[0][arg_avg]
    y_model = all_model_results[0][0][arg_avg]

    ax[0][1].set_title("avg. fit with loss = " + "{:0.3e}".format(values[0][arg_avg]))
    ax[0][1].plot(x_test, y_model, c=COLORS[2], label="model")
    ax[0][1].scatter(x_test, y_test, facecolor="white", edgecolor="black", label="test")
    ax[0][1].set_ylim(-1, 1)
    ax[0][1].yaxis.set_major_locator(FixedLocator([1, 0.5, 0, -0.5, -1]))
    ax[0][1].set_xlim(0 - 0.2, 2 * np.pi - 0.2)
    ax[0][1].tick_params(axis="both", direction="in")

    plt.tight_layout(pad=0.0)

    fig.subplots_adjust(wspace=0.45, hspace=0.05)
    fig.savefig(
        save_path + filename + ".pdf", format="pdf", dpi=1000, bbox_inches="tight"
    )


def get_coeffs_experiments_and_configs(filename):
    """Returns ansaetze and configs from stored sampled coefficients file."""
    with open(filename, "rb") as f:
        d = pickle.load(f)

    return d.get("ansaetze", None), d.get("configs", None)


def get_coeffs_model_results(filename):
    """Returns model_results, model_parameters, n_coeffs and n_samples
    from stored sampled coefficients file."""
    with open(filename, "rb") as f:
        d = pickle.load(f)

    return (
        d.get("model_results", None),
        d.get("model_parameters", None),
        d.get("n_coeffs", None),
        d.get("n_samples", None),
    )


def get_coeffs_data(coeffs_files):
    """Returns all necessary data from stored sampled coefficients file."""
    file_names = list(coeffs_files.values())

    all_ansaetze = []

    all_model_coeffs = []
    all_model_parameters = []
    all_n_coeffs = []
    all_n_samples = []

    for i, file_name in enumerate(file_names):
        file_name_path = file_name + ".pickle"

        ansaetze, _ = get_coeffs_experiments_and_configs(file_name_path)
        model_results, model_parameters, n_coeffs, n_samples = get_coeffs_model_results(
            file_name_path
        )

        all_ansaetze.append(ansaetze)

        all_model_coeffs.append(model_results)
        all_model_parameters.append(model_parameters)
        all_n_coeffs.append(n_coeffs)
        all_n_samples.append(n_samples)

    return (
        all_model_coeffs,
        all_model_parameters,
        all_ansaetze,
        all_n_coeffs,
        file_names,
    )


def determine_grid(coeffs, max_cols=8):
    """Grid (rows, columns) for plot based on number of coefficients."""
    if coeffs % max_cols == 0:
        rows = int(coeffs / max_cols)
    else:
        rows = int(coeffs / max_cols) + 1
    if coeffs >= max_cols:
        cols = max_cols
    else:
        cols = coeffs

    return rows, cols


def determine_ax(index, max_cols=8):
    row = int(index / max_cols)
    col = index % max_cols
    return row, col


def plot_sampled_coeff(
    all_coeffs,
    all_n_coeffs,
    filename="sampled_coefficients",
    save_path="./results/own/img/",
):
    """Plots sampled coefficients."""
    colors = COLORS
    n_coeffs = np.shape(all_coeffs)[3]
    fig_rows, fig_cols = determine_grid(n_coeffs)

    fig = plt.figure(figsize=(5, fig_rows))

    ax = []
    for r in range(fig_rows):
        ax_row = []
        for c in range(fig_cols):
            if c == 0:
                ax_row.append(plt.subplot2grid((fig_rows, fig_cols), (r, c)))
            else:
                ax_row.append(plt.subplot2grid((fig_rows, fig_cols), (r, c)))
        ax.append(ax_row)

    coeffs = all_coeffs[0][0]
    coeffs_real = np.real(coeffs)
    coeffs_imag = np.imag(coeffs)

    for idx in range(n_coeffs):
        row, col = determine_ax(idx)

        ax[row][col].set_title(r"$c_{}$".format("{" + str(idx) + "}"))
        ax[row][col].scatter(
            coeffs_real[:, idx],
            coeffs_imag[:, idx],
            facecolor="white",
            edgecolor=colors[5],
        )

        if idx == 0:
            ax[row][col].set_ylim(-0.5, 0.5)
            ax[row][col].set_xlim(-1, 1)
            ax[row][col].yaxis.set_major_locator(FixedLocator([-0.01, 0, 0.01]))
            ax[row][col].xaxis.set_major_locator(FixedLocator([-0.01, 0, 0.01]))
            ax[row][col].set_ylabel("|" + "{:0.0e}".format(0.01) + "|")

        elif idx <= 7:
            value = 1 / (idx**2 + 1)
            ax[row][col].set_ylim(-value, value)
            ax[row][col].set_xlim(-value, value)
            ax[row][col].yaxis.set_major_locator(FixedLocator([-0.01, 0, 0.01]))
            ax[row][col].xaxis.set_major_locator(FixedLocator([-0.01, 0, 0.01]))
        else:
            tick = 0.0001
            if idx == 8:
                ax[row][col].set_ylabel("|" + "{:0.0e}".format(tick) + "|")
            value = 1 / (2**idx)
            ax[row][col].set_ylim(-value, value)
            ax[row][col].set_xlim(-value, value)
            ax[row][col].yaxis.set_major_locator(FixedLocator([-tick, 0, tick]))
            ax[row][col].xaxis.set_major_locator(FixedLocator([-tick, 0, tick]))

        ax[row][col].tick_params(axis="both", direction="in")

    for j, ax_j in enumerate(ax):
        for i, ax_ij in enumerate(ax_j):
            plt.setp(ax_ij.get_yticklabels(), visible=False)
            plt.setp(ax_ij.get_xticklabels(), visible=False)

    fig.subplots_adjust(wspace=0.05, hspace=0.5)
    fig.savefig(
        save_path + filename + ".pdf", format="pdf", dpi=1000, bbox_inches="tight"
    )


def plot_xcorr_example(
    functions,
    norm_corr_array,
    x_train,
    x_test,
    all_coeff,
    filename="xcorr",
    save_path="./results/own/img/",
):
    """Plots cross-correlation depending on shift paramter alpha.
    Also plots the original and the one with the highest cross-correlation value."""
    alpha = np.linspace(0, 2 * np.pi, 100)

    fig = plt.figure(figsize=(10, 2))

    colors = COLORS
    color_order = [0, 1, 5]

    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax2 = plt.subplot2grid((1, 3), (0, 1))
    ax3 = plt.subplot2grid((1, 3), (0, 2), sharey=ax2)

    ax = [[ax1, ax2, ax3]]

    for f, func in enumerate(functions):
        max_corr = max(norm_corr_array[func[0]][func[2]])
        number = norm_corr_array[func[0]][func[2]].index(max_corr)
        max_shift = alpha[number]

        ax[0][0].set_title("max=" + str(round(max_corr, 3)))
        ax[0][0].scatter(
            alpha,
            norm_corr_array[func[0]][func[2]],
            c=colors[color_order[0]],
            label="max=" + str(round(max_corr, 2)),
        )
        ax[0][0].set_ylim(0, 1)
        ax[0][0].set_xlabel("shift")
        ax[0][0].set_ylabel("corr")
        xlim_offset = 0.2

        ax[0][1].set_title("org")
        ax[0][2].set_title("shifted")
        _, y_test = test_function(
            coeffs=all_coeff[func[0]], x_train=x_train, x_test=x_test, scaling=1
        )
        ax[0][1].plot(
            x_test, y_test, c=colors[color_order[0]], label="f_" + str(func[0])
        )
        ax[0][2].plot(
            x_test, y_test, c=colors[color_order[0]], label="f_" + str(func[0])
        )

        _, y_test = test_function(
            coeffs=all_coeff[func[1]], x_train=x_train, x_test=x_test, scaling=1
        )
        ax[0][1].plot(
            x_test, y_test, c=colors[color_order[1]], label="f_" + str(func[1])
        )
        ax[0][1].set_xlabel("x")

        _, y_test = test_function(
            coeffs=all_coeff[func[1]],
            x_train=x_train,
            x_test=x_test + max_shift,
            scaling=1,
        )
        ax[0][2].plot(
            x_test, y_test, c=colors[color_order[2]], label="f_" + str(func[1]) + "+"
        )
        ax[0][2].set_xlabel("x")

        ax[0][0].set_xlim(0 - xlim_offset, 2 * pi + xlim_offset)
        ax[0][0].set_ylim(0, 1)
        ax[0][0].xaxis.set_major_locator(FixedLocator([0, int(pi), int(2 * pi)]))
        ax[0][1].set_xlim(0 - xlim_offset, 2 * pi + xlim_offset)
        ax[0][1].set_ylim(-1, 1)
        ax[0][1].xaxis.set_major_locator(FixedLocator([0, int(pi), int(2 * pi)]))
        ax[0][2].set_xlim(0 - xlim_offset, 2 * pi + xlim_offset)
        ax[0][2].set_ylim(-1, 1)
        ax[0][2].xaxis.set_major_locator(FixedLocator([0, int(pi), int(2 * pi)]))

    fig.subplots_adjust(wspace=0.4, hspace=0.2)
    fig.savefig(
        save_path + filename + ".pdf", format="pdf", dpi=1000, bbox_inches="tight"
    )
