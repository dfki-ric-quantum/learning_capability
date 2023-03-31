from math import pi
import pickle
import numpy as np

from methods.utils import show_progress, import_tensorflow
tf = import_tensorflow()
from methods.run_utils import create_model

def target_function(x, scaling=1, coeffs=[0.15 + 0.15j], coeff0=0.1):
    """Generate a truncated Fourier series for given coefficients."""
    res = coeff0
    for idx, coeff in enumerate(coeffs):
        exponent = complex(0, scaling * (idx + 1) * x)
        conj_coeff = np.conjugate(coeff)
        res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
    return np.real(res)


class haltCallback(tf.keras.callbacks.Callback):
    """Callback to stop training and alter learning rate."""

    def __init__(self, finish_threshold=5e-05):
        self.finish_threshold = finish_threshold
        lr_epochs = [1 * 120, 2 * 120]
        lr_values = [0.5, 0.1, 0.05]
        self.learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            lr_epochs, lr_values
        )

    def on_epoch_end(self, epoch, logs={}):
        if logs.get("val_loss") <= self.finish_threshold:
            self.model.stop_training = True
        self.model.optimizer.lr = self.learning_rate_fn(epoch + 1)


def round_decimals_up(number: float, decimals: int = 1):
    """Returns a value rounded up to a specific number of decimal places."""
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.ceil(number)
    else:
        factor = 10**decimals
        return np.ceil(number * factor) / factor + 1 / 10 ** (decimals + 2)


def get_domain(x_steps=50):
    """Generate x-values for training and test datasets."""
    x_set = [-2 * pi * 0, 2 * pi * 1]
    n_train = int(x_steps)
    n_test = int(x_steps)
    x_train = np.linspace(x_set[0], x_set[1], n_train, endpoint=False)
    x_test = np.linspace(x_set[0], x_set[1], n_test)

    return x_train, x_test


def test_function(coeffs=None, x_train=None, x_test=None, scaling=1):
    """Generate y-values for training and test datasets."""
    coeff0 = 0
    complex_coeffs = []

    for i, _ in enumerate(coeffs):
        if i == 0:
            coeff0 = coeffs[i][0]
        else:
            complex_coeffs.append(complex(coeffs[i][0], coeffs[i][1]))

    y_train = np.array(
        [target_function(x_i, scaling, complex_coeffs, coeff0) for x_i in x_train]
    )

    y_test = np.array(
        [target_function(x_i, scaling, complex_coeffs, coeff0) for x_i in x_test]
    )

    return y_train, y_test


def load_coefficients(degree, save_path="./functions", filename="coefficients"):
    file_name = f"{save_path}/{filename}.pickle"

    with open(file_name, "rb") as f:
        d = pickle.load(f)

    return d.get("degree_" + str(degree), None)


def get_norm_rnd_function(degree, x_test, x_train):
    """Create normalized truncated Fourier series."""

    coeffs = []

    coeffs.append([np.random.randint(-50, 50) / 100])

    for i in range(int(degree)):
        coeffs.append(
            [np.random.randint(-50, 50) / 100, np.random.randint(-50, 50) / 100]
        )

    y_train, y_test = test_function(
        coeffs=coeffs, x_train=x_train, x_test=x_test, scaling=1
    )

    y_train_max = abs(max(y_train))
    y_test_max = abs(max(y_test))
    y_train_min = abs(min(y_train))
    y_test_min = abs(min(y_test))
    y_extremes = [y_train_max, y_test_max, y_train_min, y_test_min]
    y_extrem = round_decimals_up(max(y_extremes))

    coeffs_norm = []
    for coeff in coeffs:
        coeffs_norm.append(np.round(coeff / y_extrem, 3))

    y_train_norm, y_test_norm = test_function(
        coeffs=coeffs_norm, x_train=x_train, x_test=x_test, scaling=1
    )

    return y_train_norm, y_test_norm, coeffs_norm


def fit_function(quantum_model, x_train, y_train, x_test, y_test):
    """Training hyperparameters and method."""
    epochs = 360
    batch_size = 25
    lrate = 0.5

    quantum_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(lrate),
    )

    quantum_model_history = quantum_model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        callbacks=[haltCallback()],
        validation_data=(x_test, y_test),
    )

    return quantum_model_history


def produce_rnd_func(
    all_configs, degree, rnd_functions, load_coeffs, show_details=False
):
    """Main method to train truncated Fourier series.
    1) Load coefficients.
    2) Create training and test dataset.
    3) Loop over all circuit configurations.
    4) Fit each Fourier series of dataset.
    """
    all_model_results = []
    all_model_histories = []
    all_model_parameters = []
    all_y_coeffs = []
    all_y_train = []
    all_y_test = []

    x_test, x_train = get_domain(x_steps=load_coeffs.split("_")[1])

    if "False" not in load_coeffs:
        all_y_coeffs = load_coefficients(degree, filename=load_coeffs)
        if len(all_y_coeffs) < rnd_functions:
            raise ValueError(
                "More random functions requested ({})\
                then provided in pickle file ({}).".format(
                    rnd_functions, len(all_y_coeffs)
                )
            )
        else:
            all_y_coeffs = all_y_coeffs[:rnd_functions]
    else:
        if degree == 1:
            all_y_coeffs = [
                [[0], [0.2, 0.2]],
                [[0.4], [0.2, 0.2]],
            ]
        else:
            raise ValueError(
                "load_coeffs == False with degee {} not implemented".format(degree)
            )

    for y_coeffs in all_y_coeffs:
        y_train, y_test = test_function(
            coeffs=y_coeffs, x_train=x_train, x_test=x_test, scaling=1
        )
        all_y_train.append(y_train)
        all_y_test.append(y_test)

    for c, config in enumerate(all_configs):
        model_results = []
        model_histories = []

        print("%%%%%%%%%%%  Configuration  %%%%%%%%%%%")
        print(config)
        print("")
        if show_details:
            _, circuit, _, _ = create_model(**config)
            print("%%%%%%%%%%%  Circuit  %%%%%%%%%%%")
            print(circuit)
            print("")
            print("%%%%%%%%%%%  Fit functions  %%%%%%%%%%%")
            show_progress(0, "", rnd_functions, rnd_functions)
        for i, (y_train, y_test) in enumerate(zip(all_y_train, all_y_test)):
            quantum_model, circuit, _, len_trainable_symbols = create_model(**config)

            quantum_model_history = fit_function(
                quantum_model, x_train, y_train, x_test, y_test
            )

            y_results = quantum_model.predict(x_test)

            model_results.append(y_results)
            model_histories.append(quantum_model_history.history["val_loss"])

            if show_details:
                show_progress(i + 1, "", rnd_functions, rnd_functions)

        all_model_results.append(model_results)
        all_model_histories.append(model_histories)

        all_model_parameters.append(len_trainable_symbols)

    if show_details:
        print("\n", flush=True)

    return (
        x_test,
        all_y_test,
        all_y_coeffs,
        all_model_results,
        all_model_histories,
        all_model_parameters,
    )
