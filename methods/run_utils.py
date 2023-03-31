import copy
import cirq
import tensorflow as tf

from methods.quantum_layer import Input_Layer
from methods.quantum_layer import PQC_customized
from methods.quantum_circuit import create_pqc_circuit
from methods.quantum_circuit import create_dqnn_circuit
from methods.quantum_circuit import create_measurement


def gates_to_string(gates):
    """Convert cirq gates to strings."""
    gates_string = ""

    for gate in gates:
        if "rx" in str(gate):
            gates_string += "rx"
        elif "ry" in str(gate):
            gates_string += "ry"
        elif "rz" in str(gate):
            gates_string += "rz"

    return gates_string


def string_to_gates(gates_strings):
    """Convert strings to cirq gates."""
    gates = []

    for gate_string in gates_strings:
        if "rx" in str(gate_string):
            gates.append(cirq.rx)
        elif "ry" in str(gate_string):
            gates.append(cirq.ry)
        elif "rz" in str(gate_string):
            gates.append(cirq.rz)

    return gates


def get_filename(config, all_ansatz, parameters=[]):
    """Create folder and filename based on circuit configuration."""
    filename = ""

    current_ansatz = all_ansatz[0]
    if len(all_ansatz) == 1:
        filename += current_ansatz

    if current_ansatz == "DQNN" and "arch" not in parameters:
        filename += "_q" + str(config["arch"])
    elif current_ansatz != "DQNN":
        if "arch" not in parameters:
            filename += "_q" + str(config["arch"][0])
            filename += "_l" + str(config["arch"][1])

    if "zero_layer" not in parameters:
        filename += "_zl" + str(config["zero_layer"])
    if "data_reupload" not in parameters:
        filename += "_dr" + str(config["data_reupload"])
    if "entanglement_gate" not in parameters:
        filename += "_" + str(config["entanglement_gate"])
    if "entanglement_layers" not in parameters:
        filename += "_el" + str(config["entanglement_layers"])

    if current_ansatz != "DQNN":
        if "entanglement_type" not in parameters:
            filename += "_" + str(config["entanglement_type"])
        if "entanglement_structure" not in parameters:
            filename += "_" + str(config["entanglement_structure"])

    if "input_gates" not in parameters:
        filename += "_" + gates_to_string(config["input_gates"])
    if "unitary_gates" not in parameters:
        filename += "_" + gates_to_string(config["unitary_gates"])

    if "repetitions" not in parameters:
        try:
            rep = config["repetitions"]
        except:
            filename += "_None"
        else:
            filename += "_" + str(config["repetitions"])

    if parameters != []:
        for parameter in parameters:
            filename += "_" + parameter

    return filename


def define_ansatz(config, ansatz, parameter=None, parameter_type=None):
    """Convert ansatz into corresponding circuit definition
    and overwrite configuration file if necessary."""
    config_redefined = copy.deepcopy(config)

    if ansatz == "SW":
        config_redefined["zero_layer"] = False
        config_redefined["circuit_type"] = "pqc"

    elif ansatz == "WS":
        config_redefined["zero_layer"] = False
        config_redefined["circuit_type"] = "pqc_swap"

    elif ansatz == "SWW":
        config_redefined["zero_layer"] = False
        config_redefined["circuit_type"] = "pqc"
        config_redefined["entanglement_layers"] = 2

    elif ansatz == "WSW":
        config_redefined["zero_layer"] = True
        config_redefined["circuit_type"] = "pqc"

    elif ansatz == "DQNN":
        config_redefined["circuit_type"] = "dqnn"

    elif ansatz == "C14":
        config_redefined["circuit_type"] = "pqc"
        config_redefined["entanglement_type"] = "cyclic"

    else:
        raise ValueError(f"Ansatz {ansatz} is not defined.")

    if "H" in ansatz:
        config_redefined["circuit_type"] += "_H"

    if parameter is not None:
        config_redefined[parameter] = parameter_type

    return config_redefined


def get_all_configs(all_ansatz, config, parameters, parameter_types):
    """Create experiments by preparing all configuration files
    based on the given parameter list."""
    all_experiments = []
    all_configs = []
    len_param = len(parameters)
    for ansatz in all_ansatz:
        if len_param == 1:
            parameter_0 = parameters[0]

            for parameter_type_0 in parameter_types[0]:
                config_0 = define_ansatz(config, ansatz, parameter_0, parameter_type_0)
                all_configs.append(config_0)

                if parameter_0 in ("input_gates", "unitary_gates"):
                    parameter_type_0_str = gates_to_string(parameter_type_0)
                else:
                    parameter_type_0_str = parameter_type_0

                all_experiments.append([ansatz, parameter_type_0_str])

        elif len_param == 2:
            parameter_0 = parameters[0]
            parameter_1 = parameters[1]
            for parameter_type_0 in parameter_types[0]:
                for parameter_type_1 in parameter_types[1]:
                    config_0 = define_ansatz(
                        config, ansatz, parameter_0, parameter_type_0
                    )
                    config_01 = define_ansatz(
                        config_0, ansatz, parameter_1, parameter_type_1
                    )
                    all_configs.append(config_01)

                    if parameter_0 in ("input_gates", "unitary_gates"):
                        parameter_type_0_str = gates_to_string(parameter_type_0)
                    else:
                        parameter_type_0_str = parameter_type_0

                    if parameter_1 in ("input_gates", "unitary_gates"):
                        parameter_type_1_str = gates_to_string(parameter_type_1)
                    else:
                        parameter_type_1_str = parameter_type_1

                    all_experiments.append(
                        [ansatz, parameter_type_0_str, parameter_type_1_str]
                    )

        elif len_param >= 3:
            parameter_0 = parameters[0]
            parameter_1 = parameters[1]
            parameter_2 = parameters[2]

            for parameter_type_0 in parameter_types[0]:
                for parameter_type_1 in parameter_types[1]:
                    for parameter_type_2 in parameter_types[2]:
                        config_0 = define_ansatz(
                            config, ansatz, parameter_0, parameter_type_0
                        )
                        config_01 = define_ansatz(
                            config_0, ansatz, parameter_1, parameter_type_1
                        )
                        config_012 = define_ansatz(
                            config_01, ansatz, parameter_2, parameter_type_2
                        )
                        all_configs.append(config_012)

                        if parameter_0 in ("input_gates", "unitary_gates"):
                            parameter_type_0_str = gates_to_string(parameter_type_0)
                        else:
                            parameter_type_0_str = parameter_type_0

                        if parameter_1 in ("input_gates", "unitary_gates"):
                            parameter_type_1_str = gates_to_string(parameter_type_1)
                        else:
                            parameter_type_1_str = parameter_type_1

                        if parameter_2 in ("input_gates", "unitary_gates"):
                            parameter_type_2_str = gates_to_string(parameter_type_2)
                        else:
                            parameter_type_2_str = parameter_type_2

                        all_experiments.append(
                            [
                                ansatz,
                                parameter_type_0_str,
                                parameter_type_1_str,
                                parameter_type_2_str,
                            ]
                        )

    return all_experiments, all_configs


def create_model(
    circuit_type="pqc",
    arch=[3, 5],
    zero_layer=True,
    trainable_input=False,
    data_reupload=True,
    activition=None,
    entanglement_layers=1,
    entanglement_gate="CAN",
    entanglement_type="cyclic",
    entanglement_structure="strong",
    input_gates=[cirq.rx],
    unitary_gates=[cirq.ry, cirq.rz],
    n_states=1,
    repetitions=None,
    print_details=False,
):
    """Create quantum circuit and tensorflow model."""

    if circuit_type == "dqnn":
        input_symbols, trainable_symbols, circuit = create_dqnn_circuit(
            dqnn_arch=arch,
            data_reupload=data_reupload,
            trainable_input=trainable_input,
            zero_layer=zero_layer,
            circuit_type=circuit_type,
            input_gates=input_gates,
            unitary_gates=unitary_gates,
        )

        measurement = create_measurement(
            num_qubits=sum(arch), circuit_type=circuit_type
        )
    else:
        # circuit_type == 'pqc'
        input_symbols, trainable_symbols, circuit = create_pqc_circuit(
            num_qubits=arch[0],
            layers=arch[1],
            data_reupload=data_reupload,
            trainable_input=trainable_input,
            zero_layer=zero_layer,
            entanglement_layers=entanglement_layers,
            circuit_type=circuit_type,
            entanglement_gate=entanglement_gate,
            entanglement_type=entanglement_type,
            entanglement_structure=entanglement_structure,
            input_gates=input_gates,
            unitary_gates=unitary_gates,
        )
        measurement = create_measurement(num_qubits=arch[0], circuit_type=circuit_type)

    input_layer = Input_Layer(
        input_symbols=input_symbols,
        n_input=n_states,
        trainable_input=trainable_input,
        activation=activition,
        name="specific_input",
    )

    circuit_layer = PQC_customized(
        model_circuit=circuit,
        input_symbols=input_symbols,
        circuit_symbols=trainable_symbols,
        operators=measurement,
        repetitions=repetitions,
        print_details=print_details,
        name="pqc_circuit",
    )

    if print_details:
        print("########################################")
        print("Measurement:", measurement)

    input_shape = tf.keras.layers.Input(shape=(n_states,), dtype=tf.float32)

    return (
        tf.keras.Sequential([input_shape, input_layer, circuit_layer]),
        circuit,
        len(input_symbols),
        len(trainable_symbols),
    )
