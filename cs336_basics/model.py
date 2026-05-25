from __future__ import annotations

from cs336_basics import nn_utils

import math
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

EPS = 1e-5


def run_silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return in_features.sigmoid().mul(in_features)


def run_linear(
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"],
        in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:

    return torch.matmul(in_features, weights.T)


def run_embedding(
        vocab_size: int,
        d_model: int,
        weights: Float[Tensor, " vocab_size d_model"],
        token_ids: Int[Tensor, "..."],
) -> Float[Tensor, " ... d_model"]:
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

    sc = (Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        sc = sc.masked_fill(~mask, -1e9)
    sc = nn_utils.run_softmax(sc, dim=-1)
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


def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:


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


def run_rmsnorm(
        d_model: int,
        eps: float,
        weights: Float[Tensor, "d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

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


def run_transformer_block(
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
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


class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float = 10000.0,
            device=None,
            dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = torch.nn.Parameter(
            torch.empty(vocab_size, d_model, **kwargs)
        )

        self.layers = torch.nn.ModuleList()

        for i in range(num_layers):
            layer = torch.nn.ParameterDict({
                "attn_q_proj_weight": torch.nn.Parameter(torch.empty(d_model, d_model, **kwargs)),
                "attn_k_proj_weight": torch.nn.Parameter(torch.empty(d_model, d_model, **kwargs)),
                "attn_v_proj_weight": torch.nn.Parameter(torch.empty(d_model, d_model, **kwargs)),
                "attn_output_proj_weight": torch.nn.Parameter(torch.empty(d_model, d_model, **kwargs)),
                "ln1_weight": torch.nn.Parameter(torch.ones(d_model, **kwargs)),
                "ln2_weight": torch.nn.Parameter(torch.ones(d_model, **kwargs)),
                "ffn_w1_weight": torch.nn.Parameter(torch.empty(d_ff, d_model, **kwargs)),
                "ffn_w2_weight": torch.nn.Parameter(torch.empty(d_model, d_ff, **kwargs)),
                "ffn_w3_weight": torch.nn.Parameter(torch.empty(d_ff, d_model, **kwargs)),
            })
            self.layers.append(layer)

        self.ln_final_weight = torch.nn.Parameter(
            torch.ones(d_model, **kwargs)
        )

        self.lm_head_weight = torch.nn.Parameter(
            torch.empty(vocab_size, d_model, **kwargs)
        )

        self.reset_parameters()

        return

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.token_embeddings, mean=0.0, std=0.02)

        for layer in self.layers:
            torch.nn.init.trunc_normal_(layer["attn_q_proj_weight"], mean=0.0, std=0.02)
            torch.nn.init.trunc_normal_(layer["attn_k_proj_weight"], mean=0.0, std=0.02)
            torch.nn.init.trunc_normal_(layer["attn_v_proj_weight"], mean=0.0, std=0.02)
            torch.nn.init.trunc_normal_(layer["attn_output_proj_weight"], mean=0.0, std=0.02)
            torch.nn.init.trunc_normal_(layer["ffn_w1_weight"], mean=0.0, std=0.02)
            torch.nn.init.trunc_normal_(layer["ffn_w2_weight"], mean=0.0, std=0.02)
            torch.nn.init.trunc_normal_(layer["ffn_w3_weight"], mean=0.0, std=0.02)
            torch.nn.init.ones_(layer["ln1_weight"])
            torch.nn.init.ones_(layer["ln2_weight"])

        torch.nn.init.ones_(self.ln_final_weight)
        torch.nn.init.trunc_normal_(self.lm_head_weight, mean=0.0, std=0.02)

    def get_weights(self) -> dict[str, torch.Tensor]:
        weights = {
            "token_embeddings.weight": self.token_embeddings,
            "ln_final.weight": self.ln_final_weight,
            "lm_head.weight": self.lm_head_weight,
        }

        for i, layer in enumerate(self.layers):
            weights[f"layers.{i}.attn.q_proj.weight"] = layer["attn_q_proj_weight"]
            weights[f"layers.{i}.attn.k_proj.weight"] = layer["attn_k_proj_weight"]
            weights[f"layers.{i}.attn.v_proj.weight"] = layer["attn_v_proj_weight"]
            weights[f"layers.{i}.attn.output_proj.weight"] = layer["attn_output_proj_weight"]
            weights[f"layers.{i}.ln1.weight"] = layer["ln1_weight"]
            weights[f"layers.{i}.ln2.weight"] = layer["ln2_weight"]
            weights[f"layers.{i}.ffn.w1.weight"] = layer["ffn_w1_weight"]
            weights[f"layers.{i}.ffn.w2.weight"] = layer["ffn_w2_weight"]
            weights[f"layers.{i}.ffn.w3.weight"] = layer["ffn_w3_weight"]
        return weights

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self.get_weights(),
            in_indices=input_ids,
        )
