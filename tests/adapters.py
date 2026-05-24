from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import numpy.random as npr
import math
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from triton.language import dtype

import regex

EPS = 1e-5

class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        default = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, default)

    def step(self, closure: None = None):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                [beta1, beta2] = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                for param in group['params']:
                    if param.grad is None:
                        continue
                    if 'step' not in self.state[param]:
                        self.state[param]['step'] = 0
                        self.state[param]['m'] = torch.zeros_like(param)
                        self.state[param]['v'] = torch.zeros_like(param)

                    self.state[param]['step'] += 1
                    self.state[param]['m'] = beta1 * self.state[param]['m'] + (1 - beta1) * param.grad
                    self.state[param]['v'] = beta2 * self.state[param]['v'] + (1 - beta2) * torch.pow(param.grad, 2)

                    m = self.state[param]['m'] / (1 - beta1 ** self.state[param]['step'])
                    v = self.state[param]['v'] / (1 - beta2 ** self.state[param]['step'])

                    tmp = lr * weight_decay * param
                    param.sub_(tmp)
                    tmp = lr * (m / (torch.sqrt(v) + eps))
                    param.sub_(tmp)
        return

def run_linear(
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"],
        in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    return torch.matmul(in_features, weights.T)

def run_embedding(
        vocab_size: int,
        d_model: int,
        weights: Float[Tensor, " vocab_size d_model"],
        token_ids: Int[Tensor, "..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    o_shape = token_ids.shape
    ti = token_ids.reshape(-1, )
    res = torch.index_select(input=weights, dim=0, index=ti)

    return res.reshape(*o_shape, d_model)


def run_swiglu(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    W1 = torch.matmul(in_features, w1_weight.T)
    W3 = torch.matmul(in_features, w3_weight.T)
    W1 = run_silu(W1)
    return torch.matmul(W1 * W3, w2_weight.T)


def run_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    sc = (Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        sc = sc.masked_fill(~mask, -1e9)
    sc = run_softmax(sc, dim=-1)
    return sc @ V


def run_multihead_self_attention(
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_model d_model"],
        k_proj_weight: Float[Tensor, " d_model d_model"],
        v_proj_weight: Float[Tensor, " d_model d_model"],
        o_proj_weight: Float[Tensor, " d_model d_model"],
        in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """

    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    dims = d_model // num_heads
    Q = Q.reshape(*Q.shape[:-1], num_heads, dims)
    K = K.reshape(*K.shape[:-1], num_heads, dims)
    V = V.reshape(*V.shape[:-1], num_heads, dims)
    Q = Q.transpose(-2, -3)
    K = K.transpose(-2, -3)
    V = V.transpose(-2, -3)

    mask = torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=in_features.device).tril()
    res = run_scaled_dot_product_attention(Q, K, V, mask)

    res = res.transpose(-2, -3)
    res = res.reshape(*res.shape[:-2], d_model)
    return res @ o_proj_weight.T


def run_multihead_self_attention_with_rope(
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_model d_model"],
        k_proj_weight: Float[Tensor, " d_model d_model"],
        v_proj_weight: Float[Tensor, " d_model d_model"],
        o_proj_weight: Float[Tensor, " d_model d_model"],
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    if token_positions is None:
        token_positions = torch.arange(
            in_features.shape[-2],
            device=in_features.device,
        )

    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    dims = d_model // num_heads

    Q = Q.reshape(*Q.shape[:-1], num_heads, dims)
    K = K.reshape(*K.shape[:-1], num_heads, dims)
    V = V.reshape(*V.shape[:-1], num_heads, dims)
    Q = Q.transpose(-2, -3)
    K = K.transpose(-2, -3)
    V = V.transpose(-2, -3)

    Q = run_rope(dims, theta, max_seq_len, Q, token_positions)
    K = run_rope(dims, theta, max_seq_len, K, token_positions)

    mask = torch.tril(
        torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=in_features.device)
    )
    res = run_scaled_dot_product_attention(Q, K, V, mask)

    res = res.transpose(-2, -3)
    res = res.reshape(*res.shape[:-2], d_model)
    return res @ o_proj_weight.T


def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """

    o_type = in_query_or_key.dtype

    cnt = d_k // 2
    t = in_query_or_key.reshape(*in_query_or_key.shape[:-1], cnt, 2)

    j = torch.arange(
        cnt,
        device=in_query_or_key.device,
        dtype=torch.float32,
    )

    freq = 1.0 / (theta ** (2 * j / d_k))

    positions = token_positions.to(
        device=in_query_or_key.device,
        dtype=torch.float32,
    )

    angle = positions[..., None] * freq

    co = torch.cos(angle).to(o_type)
    si = torch.sin(angle).to(o_type)

    x0 = t[..., 0]
    x1 = t[..., 1]

    res0 = x0 * co - x1 * si
    res1 = x0 * si + x1 * co

    res = torch.stack([res0, res1], dim=-1)

    return res.reshape(in_query_or_key.shape)


def run_transformer_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    out = run_rmsnorm(d_model, EPS, weights['ln1.weight'], in_features)
    out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights['attn.q_proj.weight'],
        k_proj_weight=weights['attn.k_proj.weight'],
        v_proj_weight=weights['attn.v_proj.weight'],
        o_proj_weight=weights['attn.output_proj.weight'],
        in_features=out
    )
    res1 = out + in_features

    out = run_rmsnorm(d_model, EPS, weights['ln2.weight'], res1)
    out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights['ffn.w1.weight'],
        w2_weight=weights['ffn.w2.weight'],
        w3_weight=weights['ffn.w3.weight'],
        in_features=out
    )

    res2 = res1 + out

    return res2


def run_transformer_lm(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    embedding = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights['token_embeddings.weight'],
        token_ids=in_indices
    )

    layers = embedding
    for i in range(num_layers):
        block_weights = {
            "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],

            "ln1.weight": weights[f"layers.{i}.ln1.weight"],

            "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],

            "ln2.weight": weights[f"layers.{i}.ln2.weight"],
        }

        layers = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=block_weights,
            in_features=layers
        )

    rms_res = run_rmsnorm(
        d_model=d_model,
        eps=EPS,
        weights=weights['ln_final.weight'],
        in_features=layers
    )

    lm_res = rms_res @ weights['lm_head.weight'].T

    return lm_res


def run_rmsnorm(
        d_model: int,
        eps: float,
        weights: Float[Tensor, "d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """

    o_dtype = in_features.dtype
    x = in_features.to(torch.float32)
    sum2 = torch.mean(
        x * x,
        dim=-1,
        keepdim=True,
    )
    fac = torch.rsqrt(sum2 + eps)
    out = x * fac * weights
    return out.to(o_dtype)


def run_silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features.sigmoid().mul(in_features)


def run_get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    N = dataset.size - context_length
    P = npr.randint(0, N, size=batch_size)
    X = torch.empty(batch_size, context_length, dtype=torch.long, device=device)
    Y = torch.empty(batch_size, context_length, dtype=torch.long, device=device)
    cur = 0
    for pos in P:
        x = []
        y = []
        for i in range(context_length):
            x.append(dataset[pos + i])
            y.append((dataset[pos + i + 1]))
        X[cur] = torch.tensor(x, dtype=torch.long, device=device)
        Y[cur] = torch.tensor(y, dtype=torch.long, device=device)
        cur += 1
    return X, Y


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

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamWOptimizer


def run_get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    if it < warmup_iters:
        res = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        PI = math.acos(-1)
        need = cosine_cycle_iters - warmup_iters
        now = it - warmup_iters
        res = (math.cos(now / need * PI) + 1) * 0.5 * (max_learning_rate - min_learning_rate) + min_learning_rate
    else :
        res = min_learning_rate

    return res

def run_save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iteration
    }

    torch.save(checkpoint, out)


def run_load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iterations"]

class BPETokenizer:

    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.i_vocab = {}
        for key in vocab:
            val = vocab[key]
            self.i_vocab[val] = key
        self.merges = merges
        self.special_tokens: list[str] = special_tokens or []
        self.pat = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.merge_ranks = {}
        for rank, pair in enumerate(merges):
            self.merge_ranks[pair] = rank

    def merge(self, tokens):
        now = tokens

        while True:
            best_pair = None
            best_rank = None

            for i in range(len(now) - 1):
                pair = (now[i], now[i + 1])
                rank = self.merge_ranks.get(pair)

                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            nxt = []
            i = 0

            while i < len(now):
                if i < len(now) - 1 and (now[i], now[i + 1]) == best_pair:
                    nxt.append(now[i] + now[i + 1])
                    i += 2
                else:
                    nxt.append(now[i])
                    i += 1

            now = nxt

        return now

    def decode(self, ids):
        res = b""
        for token_id in ids:
            res += self.vocab[token_id]
        return res.decode("utf-8", errors="replace")

    def encode_chunk(self, text):
        tokens = [bytes([ch]) for ch in text.encode("utf-8")]
        tokens = self.merge(tokens)

        res = []
        for token in tokens:
            res.append(self.i_vocab[token])

        return res

    def encode_without_special(self, text):
        res = []
        chunks = regex.findall(self.pat, text)
        for chunk in chunks:
            res.extend(self.encode_chunk(chunk))
        return res

    def encode(self, text):
        if not self.special_tokens:
            return self.encode_without_special(text)

        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(regex.escape(tok) for tok in special_tokens) + ")"
        parts = regex.split(special_pattern, text)

        res = []
        for part in parts:
            if part == "":
                continue
            elif part in self.special_tokens:
                res.append(self.i_vocab[part.encode("utf-8")])
            else:
                res.extend(self.encode_without_special(part))

        return res

    def encode_iterable(self, iterable):
        for text in iterable:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id

def get_tokenizer(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    pat = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

        cnt = {}
        if len(special_tokens) > 0:
            spt = "|".join(regex.escape(spt) for spt in sorted(special_tokens, key=len, reverse=True))
            chunks = regex.split(f"({spt})", text)
        else:
            chunks = [text]

        sp_token = set(special_tokens)

        for chunk in chunks:
            if chunk == "" or chunk in sp_token:
                continue

            pre_tokens = regex.findall(pat, chunk)

            for pat_text in pre_tokens:
                byte_text = pat_text.encode("utf-8")
                byte_token = tuple(bytes([b]) for b in byte_text)
                if byte_token in cnt:
                    cnt[byte_token] += 1
                else:
                    cnt[byte_token] = 1

        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])
        for spt in special_tokens:
            vocab[len(vocab)] = spt.encode("utf-8")

        merges = []
        while len(vocab) < vocab_size:
            pair_cnt = {}
            for token, c in cnt.items():
                for i in range(len(token) - 1):
                    pair = (token[i], token[i + 1])

                    if pair in pair_cnt:
                        pair_cnt[pair] += c
                    else:
                        pair_cnt[pair] = c
            if len(pair_cnt) == 0:
                break
            mx_pr = max(pair_cnt, key=lambda pair: (pair_cnt[pair], pair))
            merges.append(mx_pr)

            new_token = mx_pr[0] + mx_pr[1]
            vocab[len(vocab)] = new_token

            new_cnt = {}
            for token, c in cnt.items():
                new_token = []
                i = 0
                while i < len(token):
                    if i < len(token) - 1 and mx_pr == (token[i], token[i + 1]):
                        new_token.append(token[i] + token[i + 1])
                        i += 2
                    else:
                        new_token.append(token[i])
                        i += 1
                new_token = tuple(new_token)

                if new_token in new_cnt:
                    new_cnt[new_token] += c
                else:
                    new_cnt[new_token] = c
            cnt = new_cnt

        return vocab, merges