import pickle
import numpy as np

from methods.func_utils import get_domain, get_norm_rnd_function, test_function


def save_coefficients(coef_dict, save_path, filename):
    file_name = f"{save_path}/{filename}.pickle"

    try:
        open(file_name, "rb")
        print("{} does already exist.".format(filename))
        print("No new set of coefficients created.")
        print("Use a different filename to create a new set.")
        return False

    except (OSError, IOError) as e:
        with open(file_name, "wb") as f:
            pickle.dump(coef_dict, f)
            return True


def load_coefficients(degree, save_path="./functions", filename="coefficients"):
    file_name = f"{save_path}/{filename}.pickle"

    with open(file_name, "rb") as f:
        d = pickle.load(f)

    x_test, x_train = get_domain(filename.split("_")[1])

    return d.get("degree_" + str(degree), None), x_test, x_train


def create_coefficients(
    max_degree, rnd_functions=100, filename="c_50_test", save_path="./functions"
):
    """Main method for creating coefficients."""
    x_test, x_train = get_domain(filename.split("_")[1])

    coef_dict = {}

    for d in range(1, max_degree + 1):
        d_coeffs = []

        for f in range(rnd_functions):
            y_train, y_test, y_coeffs = get_norm_rnd_function(d, x_test, x_train)

            while max(abs(y_train)) > 1.0 or max(abs(y_test)) > 1.0:
                y_train, y_test, y_coeffs = get_norm_rnd_function(d, x_test, x_train)

            d_coeffs.append(y_coeffs)

        coef_dict["degree_" + str(d)] = d_coeffs

    result=save_coefficients(coef_dict, save_path, filename)
    if not result:
        coef_dict, x_test, x_train = load_coefficients(
            max_degree, save_path, filename
        )
        print(
            f"Loaded dataset {filename} in {save_path} for degree {max_degree} instead."
        )

    return coef_dict, x_test, x_train


def corr(a, b):
    return np.dot(a, b)


def abs_corr(a, b):
    return abs(np.dot(a, b))


def norm_corr(a, b):
    return np.dot(a, b) / (np.sqrt(np.dot(np.dot(a, a), np.dot(b, b))))


def abs_norm_corr(a, b):
    return abs(np.dot(a, b) / (np.sqrt(np.dot(np.dot(a, a), np.dot(b, b)))))


def get_max_xcorr(y_base, coeff, x_train, x_test, xcorr_grid=100):
    """Determine maximal absolute cross-correlation value between two functions.
    y_base is constant and compared to function values corresponding
    to coefficients shifted by alpha.
    """
    alpha = np.linspace(0, 2 * np.pi, xcorr_grid)

    current_corr = []
    current_norm_corr = []
    for a in alpha:
        _, y_test = test_function(
            coeffs=coeff, x_train=x_train + a, x_test=x_test + a, scaling=1
        )
        current_corr.append(abs_corr(y_base, y_test))
        current_norm_corr.append(abs_norm_corr(y_base, y_test))

    return current_corr, current_norm_corr


def save_xcorr(xcorr_dict, save_path, filename):
    file_name = f"{save_path}/{filename}.pickle"

    with open(file_name, "wb") as f:
        pickle.dump(xcorr_dict, f)


def get_xcorrelation(
    all_coeff, x_train, x_test, filename="xcorr", save_path="./functions"
):
    """Main method to determine absolute, normalized cross-correlation
    for a training data set of truncated Fourier series.
    1) Load or calculate coefficients of truncated Fourier series.
    2) Determine maximal cross-correlation value for each function
    with all later functions (upper triangle).
    """
    corr_array = []
    norm_corr_array = []

    xcorr_filename = "xcorr_" + filename

    try:
        open(f"{save_path}/{xcorr_filename}.pickle")
        print(
            "No recalculation of cross-correlation for set {}.".format(filename)
        )
        print(
            "Rename file {} in folder {} to allow recalculation.".format(
                xcorr_filename, save_path
            )
        )
        corr_array, norm_corr_array = load_xcorr(xcorr_filename, save_path)
        print("Loaded previously calculated data.")

    except (OSError, IOError) as e:
        for c_base, coeff_base in enumerate(all_coeff):
            corr_array_list = []
            norm_corr_array_list = []
            _, y_base = test_function(
                coeffs=coeff_base, x_train=x_train, x_test=x_test, scaling=1
            )

            for c, coeff in enumerate(all_coeff):
                if c_base < c:
                    corr_values, norm_corr_values = get_max_xcorr(
                        y_base, coeff, x_train, x_test
                    )
                    corr_array_list.append(corr_values)
                    norm_corr_array_list.append(norm_corr_values)
            corr_array.append(corr_array_list)
            norm_corr_array.append(norm_corr_array_list)

        xcorr_dict = {}
        xcorr_dict["corr"] = corr_array
        xcorr_dict["norm_corr"] = norm_corr_array

        save_xcorr(xcorr_dict, save_path, xcorr_filename)

    return corr_array, norm_corr_array


def load_xcorr(filename="xcorr", save_path="./functions"):
    file_name = f"{save_path}/{filename}.pickle"

    with open(file_name, "rb") as f:
        d = pickle.load(f)

    return d.get("corr", None), d.get("norm_corr", None)


def get_distributions(norm_corr_array):
    """Determine distribution of absolute, normalized cross correlation values."""
    grid = 20
    distribution = [0] * (grid)
    length = 1 + 1
    step_size = length / grid

    norm_max = 0
    norm_min = 1

    data_range = np.arange(-1, 1, step_size)

    data = []
    for entry in norm_corr_array:
        for value in entry:
            max_value = max(value)
            data.append(max_value)

            if max_value > norm_max:
                norm_max = max_value

            if max_value < norm_min:
                norm_min = max_value

    data = np.sort(np.asarray(data))

    for d in reversed(data):
        for d_i, d_ref in enumerate(reversed(data_range)):
            if d >= d_ref:
                distribution[-(1 + d_i)] += 1
                break

    print("normalized min : ", norm_min)
    print("normalized mean: ", np.mean(data))
    print("normalized max : ", norm_max)
    return data_range, distribution, data


def print_distribution(data_range, distribution):
    """Print distribution"""
    print("total values: ", sum(distribution))

    r_prev = 1
    for i, (r, d) in enumerate(zip(reversed(data_range), reversed(distribution))):
        if i == 0:
            print("[{},{}]: {}".format(str(round(r_prev, 1)), str(round(r, 1)), str(d)))
        else:
            print("({},{}]: {}".format(str(round(r_prev, 1)), str(round(r, 1)), str(d)))
        r_prev = r


def extrem_functions(corr_array, data, top_nb=1):
    """Determine the function pairs top_nb bets and worst
    absolute, normalized cross-correlation values."""

    worst_xcorr = data[-top_nb:]
    best_xcorr = data[:top_nb]

    worst_functions = []
    best_functions = []
    for e, entry in enumerate(corr_array):
        for v, value in enumerate(entry):
            if np.max(value) in worst_xcorr:
                worst_functions.append([e, v + e + 1, v])
            elif np.max(value) in best_xcorr:
                best_functions.append([e, v + e + 1, v])

    return best_functions, worst_functions
