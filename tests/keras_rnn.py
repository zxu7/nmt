"""
cell: A RNN cell instance. A RNN cell is a class that has:
            - a `call(input_at_t, states_at_t)` method, returning
                `(output_at_t, states_at_t_plus_1)`.
            - a `state_size` attribute. This can be a single integer
                (single state) in which case it is
                the size of the recurrent state
                (which should be the same as the size of the cell output).
                This can also be a list/tuple of integers
                (one size per state). In this case, the first entry
                (`state_size[0]`) should be the same as
                the size of the cell output.
            It is also possible for `cell` to be a list of RNN cell instances,
            in which cases the cells get stacked on after the other in the RNN,
            implementing an efficient stacked RNN.
"""
import keras
import keras.backend as K
from keras.layers.recurrent import RNN


# First, let's define a RNN Cell, as a layer subclass.
class MinimalRNNCell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        print("DEBUG: [build] input_shape ", len(input_shape))
        print("DEBUG: [build] input_shape[0]", input_shape[0])
        print("DEBUG: [build] input_shape[1]", input_shape[1])
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        print("DEBUG: states length ", len(states))
        print("DEBUG: inputs shape ", inputs.shape)
        print("DEBUG: state shape ", states[0].shape)
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


class AttentionDecoderCell(keras.layers.Layer):
    pass


def case_rnn():
    # Let's use this cell in a RNN layer:

    cell = MinimalRNNCell(32)
    x = keras.Input((None, 5))
    layer = RNN(cell, return_sequences=True)
    # y = layer(x)

    # Here's how to use the cell to build a stacked RNN:

    # cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
    # x = keras.Input((None, 5))
    # layer = RNN(cells)
    # y = layer(x)

    inp = keras.layers.Input((5, 5))
    x = RNN(cell, return_sequences=True)(inp)
    print("DEBUG: dir ", dir(x))
    x1 = keras.layers.LSTM(3)(x)
    model = keras.models.Model(inp, x1)
    model.summary()


if __name__ == "__main__":
    case_rnn()