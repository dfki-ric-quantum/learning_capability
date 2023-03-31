import cirq
from methods.utils import import_tensorflow
tf = import_tensorflow()
import tensorflow_quantum as tfq
from math import pi


class Input_Layer(tf.keras.layers.Layer):
    def __init__(
        self, input_symbols, n_input, activation=None, trainable_input=False, **kwargs
    ):
        super().__init__(**kwargs)

        self.input_symbols = input_symbols
        self.n_input = n_input
        self.activation = activation

        self.input_parameters = self.add_weight(
            "input_parameters",
            shape=(len(self.input_symbols),),
            initializer=tf.constant_initializer(1),
            dtype=tf.float32,
            trainable=trainable_input,
        )

    def call(self, inputs):
        tensor_input = tf.convert_to_tensor(inputs, dtype=tf.float32)

        tiled_input = tf.tile(
            tensor_input, multiples=[1, int(len(self.input_symbols) / self.n_input)]
        )

        scaled_inputs = tf.einsum("i,ji->ji", self.input_parameters, tiled_input)

        all_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        return all_inputs


class PQC_customized(tf.keras.layers.Layer):
    def __init__(
        self,
        model_circuit,
        input_symbols,
        circuit_symbols,
        operators,
        repetitions=None,
        initializer=tf.keras.initializers.RandomUniform(0, 2 * pi),
        print_details=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_symbols = input_symbols
        self.circuit_symbols = circuit_symbols
        self.initializer = tf.keras.initializers.get(initializer)
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        if repetitions is None:
            self.computation_layer = tfq.layers.ControlledPQC(model_circuit, operators)
        else:
            self.computation_layer = tfq.layers.ControlledPQC(
                model_circuit,
                operators,
                repetitions=repetitions,
                backend="noiseless",
                differentiator=tfq.differentiators.ParameterShift(),
            )

        symbols = [str(symb) for symb in input_symbols + circuit_symbols]
        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols])
        if print_details:
            print("Sorted trainable parameters: \n", sorted(symbols))

        self.circuit_parameters = self.add_weight(
            "circuit_parameters",
            shape=(len(self.circuit_symbols),),
            initializer=initializer,
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs), 0)

        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)

        tiled_up_parameters = tf.tile(
            [self.circuit_parameters], multiples=[batch_dim, 1]
        )

        tiled_all_parameters = tf.concat([inputs, tiled_up_parameters], axis=1)
        joined_parameters = tf.gather(tiled_all_parameters, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_parameters])
