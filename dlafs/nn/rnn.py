from dlafs.autograd import Value
from dlafs.array import ValueArray
from dlafs.nn.common import Module, BaseNeuron, _format_activation_str


class RecurrentNeuron(BaseNeuron):

    def __init__(self, num_inputs, hidden_size, activation='tanh', neuron_id=''):
        """Initialize the weights and bias randomly, and set the activation function"""
        if neuron_id:
            neuron_id = f'_{neuron_id}'
        # Initialize the weights and bias
        self.wx = ValueArray.random_normal(shape=(num_inputs, ), label=f'wx{neuron_id}',
                                           mean=0, std=2 / (num_inputs + hidden_size))
        self.wa = ValueArray.random_normal(shape=(hidden_size, ), label=f'wa{neuron_id}',
                                           mean=0, std=1 / (hidden_size))
        self.ba = Value(0, label='ba')
        self._activation = _format_activation_str(activation)

    def __call__(self, x, a):
        """The forward pass of a single neuron"""
        if len(x) != self.wx.shape[0]:
            raise ValueError(f'Expected {self.wx.shape[0]} inputs, got {len(x)}')
        if not len(a) == self.wa.shape[0]:
            raise ValueError(f'Expected {self.wa.shape[0]} hidden inputs, got {len(a)}')
        a = ValueArray(a, label='a')

        zx = sum(wx_i * x_i for wx_i, x_i in zip(self.wx, x))
        za = sum(wa_i * a_i for wa_i, a_i in zip(self.wa, a))
        z = zx + za + self.ba

        out = self.activation(z)
        return out

    def parameters(self):
        """Return the weights and bias as a list"""
        return self.wx.values + self.wa.values + [self.ba]

    def __repr__(self):
        num_inputs = self.wx.shape[0]
        return f"RecurrentNeuron({num_inputs}, '{self._activation}')"


class RecurrentLayer(Module):

    def __init__(self, num_inputs, hidden_size, activation='tanh'):
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self._activation = activation
        self.neurons = [
            RecurrentNeuron(num_inputs, hidden_size, activation, neuron_id=i)
            for i in range(hidden_size)
        ]

    def __call__(self, x):
        """The forward pass of a single recurrent layer"""
        x = ValueArray(x)
        # Check that the number of inputs equals the number of weights
        if not x.shape[1] == self.neurons[0].wx.shape[0]:
            raise ValueError(f'Expected {self.neurons[0].wx.shape[0]} inputs, got {x.shape[1]}')

        # Initialize hidden state to zeros
        a_t = ValueArray.zeros(shape=(self.hidden_size,), label='a_t')
        a = []
        for x_t in x:
            a_t = [n(x_t, a_t) for n in self.neurons]
            a.append(a_t)
        return ValueArray(a)

    def parameters(self):
        """Return the weights and bias as a list"""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        neuron_type = str(self.neurons[0]).split('(')[0]
        return (f"RecurrentLayer({neuron_type}('{self._activation}'), "
                f"{self.num_inputs}, {self.hidden_size})")


class RecurrentNN(Module):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        """The forward pass of a recurrent NN."""
        x = ValueArray(x)
        for layer in self.layers:
            if x.dim > 1 and not isinstance(layer, RecurrentLayer):
                new_x = []
                for x_t in x:
                    new_x.append(layer(x_t))
                x = ValueArray(new_x)
            else:
                x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layers_str = ',\n  '.join([str(layer) for layer in self.layers])
        return f"RecurrentNN([\n  {layers_str}\n])"
