import random
from dlafs.autograd import Value
from dlafs.array import ValueArray


class Module:

    def zero_grad(self):
        """Reset the gradients to zero"""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class BaseNeuron(Module):

    def activation(self, input):
        if self._activation == 'tanh':
            out = input.tanh()
        elif self._activation == 'relu':
            out = input.relu()
        elif self._activation == 'sigmoid':
            out = input.sigmoid()
        else:
            out = input
        return out


class Neuron(BaseNeuron):

    def __init__(self, num_inputs, activation='tanh'):
        """Initialize the weights and bias randomly, and set the activation function"""
        self.w = ValueArray([Value(random.uniform(-1, 1), label=f'w_{i+1}') for i in range(num_inputs)])
        self.b = Value(random.uniform(-1, 1), label='b')
        self._activation = activation

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
        if self._activation == 'tanh':
            neuron_type = 'Tanh'
        elif self._activation == 'relu':
            neuron_type = 'ReLU'
        elif self._activation == 'sigmoid':
            neuron_type = 'Sigmoid'
        else:
            neuron_type = 'Linear'
        return f'{neuron_type}Neuron({len(self.w)})'


class Layer(Module):

    def __init__(self, num_inputs, num_outputs, activation='tanh'):
        self.neurons = [Neuron(num_inputs, activation) for _ in range(num_outputs)]

    def __call__(self, x):
        """The forward pass of a single layer"""
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else ValueArray(out)

    def parameters(self):
        """Return the weights and bias of the whole layer as a list"""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer({self.neurons})"


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
