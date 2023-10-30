from dlafs.array import ValueArray


def mse(y_true, y_pred):
    """Calculate the mean squared error between the true and predicted values.

    MSE is minimized when the predicted value is equal to the true value.

    MSE is defined as:
        mse = sum((y_j - pred_j)**2 for j in num_samples) / num_samples
    """
    y_true, y_pred = _coerce_single_dim_args(y_true, y_pred)
    num_items = len(y_true)
    return sum((y_i.item() - pred_i.item())**2 for y_i, pred_i in zip(y_true, y_pred)) / num_items


def accuracy(y_true, y_pred):
    """Calculate the accuracy between the true and predicted values.

    Accuracy is defined as:
        accuracy = sum(y_j == pred_j for j in num_samples) / num_samples

    The input values should be integers representing the class labels, not one-hot encoded
    vectors.
    """
    y_true, y_pred = _coerce_single_dim_args(y_true, y_pred)

    num_items = len(y_true)
    num_correct = sum((y_i == pred_i) for y_i, pred_i in zip(y_true, y_pred))
    return num_correct / num_items


def cross_entropy(y_true, y_pred):
    """Calculate the cross entropy between true and predicted values.

    Detects whether the inputs are binary or multi-class and chooses the correct cross
    entropy variant.

    The inputs should be in the shape (num_samples, num_classes), 1D inputs are converted
    to (1, num_classes).
    """
    y_true, y_pred = _coerce_multi_dim_args(y_true, y_pred)
    if y_pred.shape[1] == 1:
        # Binary classification
        loss = binary_cross_entropy(y_true, y_pred)
    else:
        # Multi-class classification
        loss = multi_cross_entropy(y_true, y_pred)
    return loss


def binary_cross_entropy(y_true, y_pred):
    """Calculate the binary cross entropy between true and predicted values.

    Binary cross entropy is minimized when the predicted value is equal to the true value.

    The inputs should be in the shape (num_samples, 1). 1D inputs are converted to
    (num_samples, 1).
    """
    y_true, y_pred = _coerce_single_dim_args(y_true, y_pred)

    loss = 0.0
    num_samples = len(y_true)
    for i in range(num_samples):
        pred_i = y_pred[i].item()
        true_i = y_true[i].item()
        sample_loss = true_i * pred_i.log() + (1 - true_i) * (1 - pred_i).log()
        sample_loss = max(sample_loss, -100)  # Prevent overflow
        loss -= sample_loss
    return loss / num_samples


def multi_cross_entropy(y_true, y_pred):
    """Calculate the categorical (multi-class) cross entropy between true and predicted values.

    Cross entropy is minimized when the predicted value of the correct class
    is 1 and the predicted value of all other classes is 0.

    The inputs should be in the shape (num_samples, num_classes). 1D inputs are converted to
    (1, num_classes).
    """
    y_true, y_pred = _coerce_multi_dim_args(y_true, y_pred)
    loss = 0.0
    num_samples = len(y_true)
    for i in range(num_samples):
        true_i = y_true[i]
        num_classes = len(true_i)
        for j in range(num_classes):
            pred_ij = y_pred[i][j].item() + 1e-50
            true_ij = true_i[j].item()
            sample_loss = true_ij * pred_ij.log()
            loss -= sample_loss
    return loss / num_samples


def _coerce_single_dim_args(y_true, y_pred):
    """Coerce the arguments to the correct types and shapes.

    1. Convert sequences to ValueArray
    2. Add extra dim to single sample
    """
    if not isinstance(y_true, ValueArray):
        y_true = ValueArray(y_true)
    if y_true.dim < 1:
        y_true = [y_true.values]
        y_true = ValueArray(y_true)

    if not isinstance(y_pred, ValueArray):
        y_pred = ValueArray(y_pred)
    if y_pred.dim < 1:
        y_pred = [y_pred.values]
        y_pred = ValueArray(y_pred)

    return y_true, y_pred


def _coerce_multi_dim_args(y_true, y_pred):
    """Coerce the arguments to the correct types and shapes.

    1. Convert sequences to ValueArray
    2. Add extra dim to single sample
    """
    if not isinstance(y_true, ValueArray):
        y_true = ValueArray(y_true)
    if y_true.dim < 2:
        y_true = [y_true.values]
        y_true = ValueArray(y_true)

    if not isinstance(y_pred, ValueArray):
        y_pred = ValueArray(y_pred)
    if y_pred.dim < 2:
        y_pred = [y_pred.values]
        y_pred = ValueArray(y_pred)

    return y_true, y_pred
