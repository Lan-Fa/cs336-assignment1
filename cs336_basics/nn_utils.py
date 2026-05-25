from collections.abc import Iterable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    mx = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - mx;
    exp = torch.exp(shifted)
    sum = torch.sum(exp, dim=dim, keepdim=True)

    return exp / sum


def run_cross_entropy(
        inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """

    mx = torch.max(input=inputs, dim=1, keepdim=True).values
    shifted = inputs - mx
    ex = torch.exp(shifted)
    loss = 0
    for i in range(inputs.shape[0]):
        sum = 0
        for j in range(inputs.shape[1]):
            sum += ex[i][j]
        loss += mx[i] + torch.log(sum) - inputs[i][targets[i]]
    return loss / targets.shape[0]


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """

    l2 = 0
    for p in parameters:
        if p.grad is not None:
            l2 += torch.sum(torch.pow(p.grad, 2))
    l2 = torch.sqrt(l2)

    if (l2 > max_l2_norm):
        for p in parameters:
            if p.grad is not None:
                p.grad *= max_l2_norm / l2