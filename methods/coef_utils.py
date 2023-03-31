import numpy as np

from methods.run_utils import create_model
from methods.utils import show_progress


def fourier_coefficients(quantum_model, K):
    """
    Computes the first 2*K+1 Fourier coefficients of a 2*pi periodic function.
    """
    n_coeffs = 2 * K + 1
    t = np.linspace(0, 2 * np.pi, n_coeffs, endpoint=False)
    y_model = np.squeeze(quantum_model.predict(t))
    y = np.fft.rfft(y_model) / t.size
    return y


def get_random_coeffs(config, n_coeffs, n_samples=100):
    """Create n_samples many samples for coefficients."""
    coeffs = []

    show_progress(0, "", n_samples, 100)
    for s in range(n_samples):
        quantum_model, _, _, _ = create_model(**config)

        coeffs_sample = fourier_coefficients(quantum_model, n_coeffs)
        coeffs.append(coeffs_sample)
        
        show_progress(s + 1, "", n_samples, 100)

    print("\n", flush=True)
    return coeffs


def produce_coeff(all_configs, degree, n_samples=100):
    """Main method for sampling coefficients.
    1) Loop over all circuit configurations.
    2) Repeatedly sample coefficients produced
       by random initialization of circuits.
    """

    all_coeffs = []
    all_model_parameters = []

    for c, config in enumerate(all_configs):

        n_coeffs = 0
        n_additional = 1
        if not (degree is None):
            n_coeffs = degree + n_additional
        elif config["circuit_type"] == "dqnn":
            if config["data_reupload"] == False:
                n_coeffs = config["n_qubits"][0] + 1 + n_additional
            else:
                n_coeffs = sum(config["n_qubits"][:-1]) + 1 + n_additional
        else:
            n_coeffs = config["n_qubits"] * config["layers"] + 1 + n_additional

        print("%%%%%%%%%%%  Sample coefficients  %%%%%%%%%%%")
        model_coeffs = get_random_coeffs(config, n_coeffs, n_samples=n_samples)

        all_coeffs.append(model_coeffs)

        _, _, _, len_trainable_symbols = create_model(**config)
        all_model_parameters.append(len_trainable_symbols)

    return all_coeffs, all_model_parameters, n_coeffs, n_samples
