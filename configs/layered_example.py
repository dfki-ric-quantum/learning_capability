import cirq

ANSATZ = ["WSW"]

FUNC_CONFIG = dict(
    degree = 12,
    load_coefficients = 'c_100_a',
    n_functions = 100,
)

SAMPLE_CONFIG = dict(
    n_samples = 100,
)

QUANTUM_CONFIG = dict(
    arch=[4,3],
    zero_layer=True,
    trainable_input=False,
    data_reupload=True,
    activition=None,
    entanglement_layers=3,
    entanglement_gate='CRX', 
    entanglement_type='linear',
    entanglement_structure='simple',
    input_gates = [cirq.rx],
    unitary_gates = [cirq.ry, cirq.rz],
    repetitions = None,
)
