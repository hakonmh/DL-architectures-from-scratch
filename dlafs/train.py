class Trainer:

    def __init__(self, model, loss, learning_rate):
        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate

    def train(self, inputs, labels, num_iterations):
        for i in range(num_iterations):
            outputs = (self.model(xi) for xi in inputs)
            loss = self.loss(labels, outputs)
            loss.backward()
            update_weights(self.model, learning_rate=self.learning_rate)
            print(f'{i}: {loss.data:.4f}')


def update_weights(model, learning_rate=1e-2):
    for parameter in model.parameters():
        parameter.data -= parameter.grad * learning_rate
    model.zero_grad()  # Reset the gradients to zero
    return model
