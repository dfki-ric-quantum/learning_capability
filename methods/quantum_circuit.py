import cirq
import sympy
import tensorflow_quantum as tfq
import numpy as np


def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a


def flatten_tuple(nested_tuple):
    flatten_tuple = ()

    for sublist in nested_tuple:
        if type(sublist) == tuple:
            for item in sublist:
                flatten_tuple += (item,)
        else:
            flatten_tuple += (sublist,)

    return flatten_tuple


def get_new_param(symbol_name, qubit, position, layer=None):
    """
    Return new learnable parameter
    """
    if str(layer):
        new_param = sympy.symbols(
            symbol_name + "_" + str(qubit) + "_" + str(layer) + "_" + str(position)
        )
    else:
        new_param = sympy.symbols(symbol_name + "_" + str(qubit) + "_" + str(position))

    return new_param


def create_measurement(num_qubits=1, circuit_type="pqc", measurement_position=-1):
    qubits = cirq.GridQubit.rect(1, num_qubits)
    measurement = []

    if circuit_type == "dqnn":
        measurement.append(cirq.Z(qubits[measurement_position]))
    elif "pqc" in circuit_type:
        measurement.append(cirq.Z(qubits[measurement_position]))
    else:
        raise ValueError("circuit_type {} is not defined.".format(circuit_type))

    return measurement


def create_input(
    qubit, n_qubit, symbol_name, layer=None, circuit_type="pqc", input_gates=[cirq.rx]
):
    circuit = cirq.Circuit()

    input_parameters = []

    for i, gate in enumerate(input_gates):
        input_parameter = get_new_param(symbol_name, n_qubit, i, layer)
        input_parameters.append(input_parameter)
        circuit += gate(input_parameter).on(qubit)

    return circuit, to_tuple(input_parameters)


def create_unitary(
    qubit,
    n_qubit,
    layer,
    rot_counter=0,
    circuit_type="pqc",
    unitary_gates=[cirq.ry, cirq.rz],
):
    circuit = cirq.Circuit()
    body_parameters = []
    symbol_name = "train"
    for i, gate in enumerate(unitary_gates):
        body_parameter = get_new_param(symbol_name, n_qubit, rot_counter + i, layer)
        body_parameters.append(body_parameter)
        circuit += gate(body_parameter).on(qubit)

    return circuit, to_tuple(body_parameters)


def create_can(target_qubit, control_qubit, qubit, position, layer):
    """
    According to implementation of class MSGate(ops.XXPowGate).
    """
    circuit = cirq.Circuit()
    can_parameters = []
    symbol_name = "train_can"

    can_parameter = get_new_param(symbol_name, qubit, position + 0, layer)
    circuit += cirq.ops.XXPowGate(
        exponent=can_parameter * 2 / np.pi, global_shift=-0.5
    ).on(target_qubit, control_qubit)
    can_parameters.append(can_parameter)

    can_parameter = get_new_param(symbol_name, qubit, position + 1, layer)
    circuit += cirq.ops.YYPowGate(
        exponent=can_parameter * 2 / np.pi, global_shift=-0.5
    ).on(target_qubit, control_qubit)
    can_parameters.append(can_parameter)

    can_parameter = get_new_param(symbol_name, qubit, position + 2, layer)
    circuit += cirq.ops.ZZPowGate(
        exponent=can_parameter * 2 / np.pi, global_shift=-0.5
    ).on(target_qubit, control_qubit)
    can_parameters.append(can_parameter)

    return circuit, to_tuple(can_parameters)


def select_entanglement_operation(
    entanglement_gate,
    target_qubit,
    control_qubit,
    symbol_name=None,
    qubit=None,
    position=None,
    layer=None,
):
    entanglement_parameters = []

    if entanglement_gate == "CNOT":
        operation = cirq.CNOT(control_qubit, target_qubit)

    elif entanglement_gate == "CZ":
        operation = cirq.CZ(control_qubit, target_qubit)

    elif entanglement_gate == "CRX":
        entanglement_parameter = get_new_param(symbol_name, qubit, position, layer)
        operation = (
            cirq.rx(entanglement_parameter)
            .on(target_qubit)
            .controlled_by(control_qubit)
        )
        entanglement_parameters.append(entanglement_parameter)

    elif entanglement_gate == "CRZ":
        entanglement_parameter = get_new_param(symbol_name, qubit, position, layer)
        operation = (
            cirq.rz(entanglement_parameter)
            .on(target_qubit)
            .controlled_by(control_qubit)
        )
        entanglement_parameters.append(entanglement_parameter)

    elif entanglement_gate == "CAN":
        circuit_can, can_parameters = create_can(
            target_qubit, control_qubit, qubit, 3 * position, layer
        )
        operation = circuit_can
        entanglement_parameters.append(can_parameters)

    else:
        raise ValueError("entanglement_gate {} not defined.".format(entanglement_gate))

    return operation, to_tuple(entanglement_parameters)


def create_entanglement(
    num_qubits,
    layer,
    entanglement_layer,
    circuit_type="pqc",
    entanglement_gate="CNOT",
    entanglement_type="cyclic",
    entanglement_structure="strong",
):
    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, num_qubits)

    symbol_name = "train_ent"
    entanglement_parameters = ()

    strong_range = [0, num_qubits - 2]

    if num_qubits == 1:
        return circuit, entanglement_parameters

    elif num_qubits == 2:
        if entanglement_structure in ("strong", "strongc14", "alt"):
            target_position = (entanglement_layer + 1) % 2
            control_position = (entanglement_layer) % 2

        else:
            target_position = 1
            control_position = 0

        operation, entanglement_parameter = select_entanglement_operation(
            entanglement_gate=entanglement_gate,
            target_qubit=qubits[target_position],
            control_qubit=qubits[control_position],
            symbol_name=symbol_name,
            qubit=target_position,
            position=entanglement_layer,
            layer=layer,
        )

        circuit += operation
        entanglement_parameters += entanglement_parameter

    else:
        if "cyclic" in entanglement_type:
            qubit_range = range(num_qubits)
        else:
            qubit_range = range(num_qubits - 1)

        for i in qubit_range:
            if entanglement_structure == "strong":
                if (
                    num_qubits % 2 == 0
                    and (entanglement_layer % (num_qubits - 1) + 1)
                    % int(num_qubits / 2)
                    == 0
                    and i == int(num_qubits / 2)
                ):
                    j_t = i
                    j_c = i + 1
                else:
                    j_t = i
                    j_c = i
                target_position = (
                    j_t + entanglement_layer % (num_qubits - 1) + 1
                ) % num_qubits
                control_position = j_c % num_qubits

            elif entanglement_structure == "strongc14":
                j = i

                control_position = (
                    (-1) ** (entanglement_layer + 1) * j + num_qubits - 1
                ) % num_qubits
                target_position = (
                    (-1) ** (entanglement_layer + 1) * j
                    + strong_range[entanglement_layer % 2]
                ) % num_qubits

            elif entanglement_structure == "alt" and (i + entanglement_layer) % 2 == 1:
                continue

            else:
                control_position = i
                target_position = (i + 1) % num_qubits

            operation, entanglement_parameter = select_entanglement_operation(
                entanglement_gate=entanglement_gate,
                target_qubit=qubits[target_position],
                control_qubit=qubits[control_position],
                symbol_name=symbol_name,
                qubit=target_position,
                position=entanglement_layer,
                layer=layer,
            )

            circuit += operation
            entanglement_parameters += entanglement_parameter

    return circuit, to_tuple(entanglement_parameters)


def create_parameterized_layer(
    num_qubits,
    circuit_type,
    layer,
    entanglement_layers=1,
    entanglement_gate="CNOT",
    entanglement_type="cyclic",
    entanglement_structure="strong",
    input_gates=[cirq.rx],
    unitary_gates=[cirq.ry, cirq.rz],
):
    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, num_qubits)

    parameterized_symbols = ()

    for e_l in range(entanglement_layers):
        if "own" in circuit_type:
            circuit_ent, entanglement_parameters = create_entanglement(
                num_qubits,
                layer,
                e_l,
                circuit_type=circuit_type,
                entanglement_gate=entanglement_gate,
                entanglement_type=entanglement_type,
                entanglement_structure=entanglement_structure,
            )
            parameterized_symbols += entanglement_parameters
            circuit += circuit_ent

        for i, qubit in enumerate(qubits):
            circuit_unitary, unitary_parameters = create_unitary(
                qubit, i, layer, e_l * len(unitary_gates), circuit_type, unitary_gates
            )
            parameterized_symbols += unitary_parameters
            circuit += circuit_unitary

        circuit_ent, entanglement_parameters = create_entanglement(
            num_qubits,
            layer,
            e_l,
            circuit_type=circuit_type,
            entanglement_gate=entanglement_gate,
            entanglement_type=entanglement_type,
            entanglement_structure=entanglement_structure,
        )
        parameterized_symbols += entanglement_parameters
        circuit += circuit_ent

    parameterized_symbols = flatten_tuple(parameterized_symbols)

    return circuit, parameterized_symbols


def create_pqc_circuit(
    num_qubits,
    layers,
    data_reupload=True,
    trainable_input=False,
    zero_layer=True,
    entanglement_layers=1,
    circuit_type="pqc",
    entanglement_gate="CNOT",
    entanglement_type="cyclic",
    entanglement_structure="strong",
    input_gates=[cirq.rx],
    unitary_gates=[cirq.ry, cirq.rz],
):
    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(1, num_qubits)

    input_symbols_name = "in"
    if trainable_input:
        input_symbols_name = "in_train"

    input_symbols = ()
    trainable_symbols = ()

    for l in range(layers + 1):
        if l == 0 and "_H" in circuit_type:
            for i, qubit in enumerate(qubits):
                circuit += cirq.H(qubit)

        if l == 0 and zero_layer:
            parameterized_circuit, parameterized_symbols = create_parameterized_layer(
                num_qubits,
                circuit_type,
                l,
                entanglement_layers=entanglement_layers,
                entanglement_gate=entanglement_gate,
                entanglement_type=entanglement_type,
                entanglement_structure=entanglement_structure,
                input_gates=input_gates,
                unitary_gates=unitary_gates,
            )

            circuit += parameterized_circuit
            trainable_symbols += parameterized_symbols

        elif l > 0:
            if l == 1 or data_reupload:
                for i, qubit in enumerate(qubits):
                    input_circuit, input_parameters = create_input(
                        qubit, i, input_symbols_name, l, circuit_type, input_gates
                    )
                    circuit += input_circuit
                    input_symbols += input_parameters

            if not ("swap" in circuit_type):
                (
                    parameterized_circuit,
                    parameterized_symbols,
                ) = create_parameterized_layer(
                    num_qubits,
                    circuit_type,
                    l,
                    entanglement_layers=entanglement_layers,
                    entanglement_gate=entanglement_gate,
                    entanglement_type=entanglement_type,
                    entanglement_structure=entanglement_structure,
                    input_gates=input_gates,
                    unitary_gates=unitary_gates,
                )

                circuit += parameterized_circuit
                trainable_symbols += parameterized_symbols

    return input_symbols, trainable_symbols, circuit


def create_dqnn_circuit(
    dqnn_arch=[1, 1],
    data_reupload=True,
    trainable_input=False,
    zero_layer=False,
    circuit_type="dqnn",
    input_gates=[cirq.rx],
    unitary_gates=[cirq.ry, cirq.rz],
):
    circuit = cirq.Circuit()
    n_qubits = sum(dqnn_arch)
    qubits = cirq.GridQubit.rect(1, n_qubits)

    input_symbols_name = "in"
    if trainable_input:
        input_symbols_name = "in_train"

    input_symbols = ()
    parameterized_symbols = ()
    dqnn_previous_l = 0
    dqnn_previous_n = 0

    for l, dqnn_l in enumerate(dqnn_arch):
        # First layer
        if l == 0:
            for q_i in range(dqnn_l):
                e_l = 0
                if zero_layer:
                    circuit_unitary, unitary_parameters = create_unitary(
                        qubits[q_i],
                        q_i,
                        l,
                        e_l * len(unitary_gates),
                        circuit_type,
                        unitary_gates,
                    )
                    parameterized_symbols += unitary_parameters
                    circuit += circuit_unitary

                input_circuit, input_parameters = create_input(
                    qubits[q_i], q_i, input_symbols_name, l, circuit_type, input_gates
                )
                circuit += input_circuit
                input_symbols += input_parameters

                circuit_unitary, unitary_parameters = create_unitary(
                    qubits[q_i],
                    q_i,
                    l,
                    (e_l + zero_layer) * len(unitary_gates),
                    circuit_type,
                    unitary_gates,
                )
                circuit += circuit_unitary
                parameterized_symbols += unitary_parameters

            dqnn_previous_l = dqnn_l
        else:
            # Entangle
            for q_i in range(dqnn_previous_l):
                for q_j in range(dqnn_l):
                    circuit_can, can_parameters = create_can(
                        qubits[q_i + dqnn_previous_n],
                        qubits[q_j + dqnn_previous_n + dqnn_previous_l],
                        q_j + dqnn_previous_n + dqnn_previous_l,
                        0,
                        q_i + dqnn_previous_n,
                    )
                    circuit += circuit_can
                    parameterized_symbols += can_parameters

            # n layer
            for q_i in range(dqnn_l):
                if zero_layer and data_reupload:
                    circuit_unitary, unitary_parameters = create_unitary(
                        qubits[q_i + dqnn_previous_n + dqnn_previous_l],
                        q_i + dqnn_previous_n + dqnn_previous_l,
                        l,
                        e_l * len(unitary_gates),
                        circuit_type,
                        unitary_gates,
                    )
                    parameterized_symbols += unitary_parameters
                    circuit += circuit_unitary

                if data_reupload and not (l == len(dqnn_arch) - 1):
                    input_circuit, input_parameters = create_input(
                        qubits[q_i + dqnn_previous_n + dqnn_previous_l],
                        q_i + dqnn_previous_n + dqnn_previous_l,
                        input_symbols_name,
                        l,
                        circuit_type,
                        input_gates,
                    )
                    circuit += input_circuit
                    input_symbols += input_parameters

                circuit_unitary, unitary_parameters = create_unitary(
                    qubits[q_i + dqnn_previous_n + dqnn_previous_l],
                    q_i + dqnn_previous_n + dqnn_previous_l,
                    l,
                    (e_l + (zero_layer and data_reupload)) * len(unitary_gates),
                    circuit_type,
                    unitary_gates,
                )
                circuit += circuit_unitary
                parameterized_symbols += unitary_parameters

            dqnn_previous_n += dqnn_previous_l
            dqnn_previous_l = dqnn_l

    return input_symbols, parameterized_symbols, circuit
