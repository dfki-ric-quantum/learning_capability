from methods.utils import save_results, save_sampled_coefficients, create_results_dir
from methods.run_utils import get_filename, get_all_configs

from methods.coef_utils import produce_coeff
from methods.func_utils import produce_rnd_func
from methods.plot_utils import get_data, plot_lc_with_avg
from methods.plot_utils import get_coeffs_data, plot_sampled_coeff


def run_experiments(
    ansatz, quantum_config, func_config, sample_config, save_path="./results/own/"
):
    func_filename = ""
    coeff_filename = ""
    # load config

    degree = func_config["degree"]
    n_functions = func_config["n_functions"]
    load_coefficients = func_config["load_coefficients"]

    parameters = ["entanglement_gate"]
    parameter_types = [[quantum_config["entanglement_gate"]]]

    # define experiements
    experiments, configs = get_all_configs(
        ansatz, quantum_config, parameters, parameter_types
    )

    # create filename and output directory
    filename = get_filename(quantum_config, ansatz, parameters)
    save_file_path = save_path + filename
    create_results_dir(save_file_path)

    # fit functions
    (
        x_test,
        y_test,
        coeffs,
        model_results,
        model_histories,
        model_parameters,
    ) = produce_rnd_func(
        configs, degree, n_functions, load_coefficients, show_details=True
    )

    # save training results
    func_filename = save_results(
        load_coefficients,
        experiments,
        configs,
        x_test,
        y_test,
        coeffs,
        model_results,
        model_parameters,
        model_histories,
        save_file_path,
        filename,
    )

    # plot lc
    func_files = {}
    value = func_filename.replace(".pickle", "")
    key = filename
    func_files[key] = value

    # load function data
    (
        _,
        _,
        all_model_histories_last,
        _,
        _,
        _,
        _,
        x_test,
        all_y_test,
        all_model_results,
    ) = get_data(func_files)

    # plot function
    plot_lc_with_avg(x_test, all_y_test, all_model_results, all_model_histories_last)

    # sample coefficients
    n_samples = sample_config["n_samples"]
    if n_samples > 0:
        # sample coefficients
        model_results, model_parameters, n_coeffs, _ = produce_coeff(
            configs, degree=degree, n_samples=n_samples
        )

        # save sampled coefficients
        coeffs_filename = save_sampled_coefficients(
            experiments,
            configs,
            model_results,
            model_parameters,
            n_coeffs,
            n_samples,
            save_file_path,
            filename,
        )

        # plot sampled coefficients
        coeffs_files = {}
        value = coeffs_filename.replace(".pickle", "")
        key = filename
        coeffs_files[key] = value
        # load sampled coeffecients
        all_model_coeffs, _, _, all_n_coeffs, _ = get_coeffs_data(coeffs_files)

        # plot sampled coefficients
        plot_sampled_coeff(all_model_coeffs, all_n_coeffs)

    return func_filename, coeff_filename
