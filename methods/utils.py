import os
import argparse
import pickle
from datetime import datetime
import sys

def import_tensorflow():
    # Filter tensorflow version warnings
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf
tf = import_tensorflow()


def set_gpu_memory_growing():
    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (ValueError, RuntimeError) as err:
            print(f"Can't set memory growth: {err}")


def show_progress(index, prefix="", max_range=100, max_size=100, out=sys.stdout):
    x = int(max_size * index / max_range)
    print(
        "{}[{}{}] {}/{}".format(
            prefix, "#" * x, "." * (max_size - x), index, max_range
        ),
        flush=True,
        end="\r",
        file=out,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Function fit and coefficients")
    parser.add_argument("-c", type=str, required=True, dest="config")
    parser.add_argument("-f", type=str, default="False", required=False, dest="func")
    parser.add_argument("-s", type=int, default=0, required=False, dest="n_samples")

    return parser.parse_args()


def create_results_dir(dirname):
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError as err:
        raise RuntimeError(f"Could not create target directories: {err}")


def save_results(
    coeffs_file,
    ansaetze,
    configs,
    x_test,
    y_test,
    coeffs,
    model_results,
    model_parameters,
    model_histories,
    save_path,
    filename,
):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    res = {
        "time": time,
        "coeffs_file": coeffs_file,
        "ansaetze": ansaetze,
        "configs": configs,
        "x_test": x_test,
        "y_test": y_test,
        "coeffs": coeffs,
        "model_results": model_results,
        "model_parameters": model_parameters,
        "model_histories": model_histories,
    }

    file_name = f"{save_path}/functs_{filename}_run_{time}.pickle"

    with open(file_name, "wb") as f:
        pickle.dump(res, f)

    print("%%%%%%%%%%%  Saved to  %%%%%%%%%%%\n", file_name, "\n")
    return file_name


def save_sampled_coefficients(
    ansaetze,
    configs,
    model_results,
    model_parameters,
    n_coeffs,
    n_samples,
    save_path,
    filename,
):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    res = {
        "time": time,
        "ansaetze": ansaetze,
        "configs": configs,
        "model_results": model_results,
        "model_parameters": model_parameters,
        "n_samples": n_samples,
        "n_coeffs": n_coeffs,
    }

    file_name = f"{save_path}/coeffs_{filename}_run_{time}.pickle"

    with open(file_name, "wb") as f:
        pickle.dump(res, f)

    print("%%%%%%%%%%%  Saved to  %%%%%%%%%%%\n", file_name, "\n")
    return file_name
