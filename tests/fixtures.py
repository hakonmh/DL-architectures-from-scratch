import math
import torch


def assert_grads_equal_expected(values, expected_grads):
    for value, exp_grad in zip(values, expected_grads):
        if isinstance(exp_grad, torch.Tensor):
            exp_grad = exp_grad.grad.item()
        assert math.isclose(value.grad, exp_grad)
