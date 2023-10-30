class Module:

    def zero_grad(self):
        """Reset the gradients to zero"""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class BaseNeuron(Module):

    def activation(self, input):
        if self._activation == 'Tanh':
            out = input.tanh()
        elif self._activation == 'ReLU':
            out = input.relu()
        elif self._activation == 'Sigmoid':
            out = input.sigmoid()
        else:
            out = input
        return out


def _format_activation_str(activation):
    """Formats the activation function as a string."""
    if activation.lower() == 'tanh':
        activation = 'Tanh'
    elif activation.lower() == 'relu':
        activation = 'ReLU'
    elif activation.lower() == 'sigmoid':
        activation = 'Sigmoid'
    else:
        activation = 'Linear'
    return activation
