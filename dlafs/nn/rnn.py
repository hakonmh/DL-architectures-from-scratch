from dlafs.autograd import Value
from dlafs.array import ValueArray
from dlafs.nn.dnn import Module, BaseNeuron


class RecurrentNeuron(BaseNeuron):

    def __init__(self, num_inputs, hidden_size, activation='tanh', neuron_id=''):
        if neuron_id:
            neuron_id = f'_{neuron_id}'
        # Initialize the weights and bias
        self.wx = ValueArray.random_normal(shape=(num_inputs, ), label=f'wx{neuron_id}',
                                           mean=0, std=2 / (num_inputs + hidden_size))
        self.wa = ValueArray.random_normal(shape=(hidden_size, ), label=f'wa{neuron_id}',
                                           mean=0, std=1 / (hidden_size))
        self.ba = Value(0, label='ba')
        self._activation = activation

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
        return f'RNN-Neuron({self.wx.shape[0]})'


class RecurrentLayer(Module):

    def __init__(self, num_inputs, hidden_size, activation='tanh'):
        self.hidden_size = hidden_size
        self.neurons = [RecurrentNeuron(num_inputs, hidden_size, activation, neuron_id=i) for i in range(hidden_size)]

    def __call__(self, x):
        x = ValueArray(x)
        # Check that the number of inputs equals the number of weights
        if not x.shape[1] == self.neurons[0].wx.shape[0]:
            raise ValueError(f'Expected {self.neurons[0].wx.shape[0]} inputs, got {x.shape[1]}')

        a_t = ValueArray.zeros(shape=(self.hidden_size,), label='a_t')  # Initialize hidden state to zeros
        a = []
        for x_t in x:
            a_t = [n(x_t, a_t) for n in self.neurons]
            a.append(a_t)
        return ValueArray(a)

    def parameters(self):
        """Return the weights and bias as a list"""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"RecurrentLayer({self.neurons})"


class RecurrentNN(Module):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        x = ValueArray(x)
        for layer in self.layers:
            if x.dim() > 1 and not isinstance(layer, RecurrentLayer):
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
