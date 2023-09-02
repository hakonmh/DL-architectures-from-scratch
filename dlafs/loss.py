import math


def mse(y_true, y_pred):
    """Calculate the mean squared error between the true and predicted values.

    MSE is minimized when the predicted value is equal to the true value.

    MSE is defined as:
        mse = sum((y_j - pred_j)**2 for j in num_samples) / num_samples
    """
    # Convert inputs to lists of values if they are scalars
    if not hasattr(y_true, '__iter__'):
        y_true = [y_true]
    if not hasattr(y_pred, '__iter__'):
        y_pred = [y_pred]

    num_items = len(y_true)
    return sum((y_i - pred_i)**2 for y_i, pred_i in zip(y_true, y_pred)) / num_items


def accuracy(y_true, y_pred):
    """Calculate the accuracy between the true and predicted values.

    Accuracy is defined as:
        accuracy = sum(y_j == pred_j for j in num_samples) / num_samples

    The input values should be integers representing the class labels, not one-hot encoded
    vectors.
    """
    # Convert inputs to lists of values if they are scalars
    if not hasattr(y_true, '__iter__'):
        y_true = [y_true]
    if not hasattr(y_pred, '__iter__'):
        y_pred = [y_pred]

    num_items = len(y_true)
    num_correct = sum((y_i == pred_i) for y_i, pred_i in zip(y_true, y_pred))
    return num_correct / num_items


def cross_entropy(y_true, y_pred):
    """Calculate the categorical (multi-class) cross entropy between true and predicted values.

    Cross entropy is minimized when the predicted value of the correct class
    is 1 and the predicted value of all other classes is 0.

    The cross entropy loss for one sample is defined as:
        h(y, y_pred) = -sum(y_i * log(pred_i) for i in num_classes)
    The cross entropy loss for multiple samples it is defined as:
        cross_entropy = sum(h(y_j, y_pred_j) for j in num_samples) / num_samples
    """
    # Inputs should be a list of lists, where each inner list is one sample.
    if not hasattr(y_true[0], '__iter__'):
        y_true = [y_true]
    if not hasattr(y_pred[0], '__iter__'):
        y_pred = [y_pred]

    EPSILON = 1e-15
    loss = 0.0
    num_samples = len(y_true)
    for i in range(num_samples):
        num_classes = len(y_true[i])
        for j in range(num_classes):
            pred_ij = y_pred[i][j]
            try:
                # Add epsilon to avoid log(0)
                pred_ij = pred_ij if pred_ij.data > EPSILON else pred_ij + EPSILON
                loss += -y_true[i][j] * pred_ij.log()
            except AttributeError:  # Values are not Value objects
                # Add epsilon to avoid log(0)
                pred_ij = max(pred_ij, EPSILON)
                loss += -y_true[i][j] * math.log(pred_ij)
    return loss / num_samples
