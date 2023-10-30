import random
from dlafs.autograd import Value
from dlafs.array import ValueArray
from dlafs.nn.common import Module, BaseNeuron, _format_activation_str


class Neuron(BaseNeuron):

    def __init__(self, num_inputs, activation='tanh', neuron_id=''):
        """Initialize the weights and bias randomly, and set the activation function"""
        if neuron_id is not None:
            neuron_id = f'_{neuron_id}'

        self.w = ValueArray.random_uniform((num_inputs, ), low=-1, high=1, label=f'w{neuron_id}')
        self.b = Value(random.uniform(-1, 1), label=f'b{neuron_id}')
        self._activation = _format_activation_str(activation)

    def __call__(self, x):
        """The forward pass of a single neuron"""
        # Check that the number of inputs equals the number of weights
        if len(x) != len(self.w):
            raise ValueError(f'Expected {len(self.w)} inputs, got {len(x)}')

        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = self.activation(z)
        return out

    def parameters(self):
        """Return the weights and bias as a list"""
        return self.w.values + [self.b]

    def __repr__(self):
        num_inputs = self.w.shape[0]
        return f"StandardNeuron({num_inputs}, '{self._activation}')"


class Layer(Module):

    def __init__(self, num_inputs, num_outputs, activation='tanh'):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._activation = activation
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_outputs)]

    def __call__(self, x):
        """The forward pass of a single layer"""
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else ValueArray(out)

    def parameters(self):
        """Return the weights and bias of the whole layer as a list"""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        neuron_type = str(self.neurons[0]).split('(')[0]
        return (f"Layer({neuron_type}('{self._activation}'), "
                f"{self.num_inputs}, {self.num_outputs})")


class VanillaNN(Module):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        """The forward pass of a full network"""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Return the weights and bias of the whole network as a list"""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layers_str = ',\n  '.join([str(layer) for layer in self.layers])
        return f"VanillaNN([\n  {layers_str}\n])"
