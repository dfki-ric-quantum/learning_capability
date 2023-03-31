import cirq

ANSATZ = ["DQNN"]

FUNC_CONFIG = dict(
    degree = 6,
    load_coefficients = 'c_50_a',
    n_functions = 100,
)

SAMPLE_CONFIG = dict(
    n_samples = 100
)

QUANTUM_CONFIG = dict(
    arch=[2,2,2,1],
    zero_layer=True,
    trainable_input=False,
    data_reupload=True,
    activition=None,
    entanglement_layers=1,
    entanglement_gate='CAN', 
    input_gates = [cirq.rx],
    unitary_gates = [cirq.ry, cirq.rz],
    repetitions = None,
)
