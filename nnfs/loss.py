def mse(y_true, y_pred):
    # Convert inputs to lists of values if they are scalars
    if not hasattr(y_true, '__iter__'):
        y_true = [y_true]
    if not hasattr(y_pred, '__iter__'):
        y_pred = [y_pred]

    num_items = len(y_true)
    return sum((y_i - pred_i)**2 for y_i, pred_i in zip(y_true, y_pred)) / num_items


def accuracy(y_true, y_pred):
    # Convert inputs to lists of values if they are scalars
    if not hasattr(y_true, '__iter__'):
        y_true = [y_true]
    if not hasattr(y_pred, '__iter__'):
        y_pred = [y_pred]

    num_items = len(y_true)
    return sum((y_i == pred_i) for y_i, pred_i in zip(y_true, y_pred)) / num_items
