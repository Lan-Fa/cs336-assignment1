import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import get_tokenizer


def load_tokenizer(vocab_path, merges_path, special_tokens):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    with open(merges_path, "rb") as f:
        merges = pickle.load(f)

    return get_tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )


def load_model(args, device):
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        iteration = checkpoint.get("iterations", None)
    else:
        state_dict = checkpoint
        iteration = None

    model.load_state_dict(state_dict)
    model.eval()

    if iteration is not None:
        print(f"loaded checkpoint: {args.checkpoint}, iteration={iteration}")
    else:
        print(f"loaded checkpoint: {args.checkpoint}")

    return model


def top_p_sample(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    keep = cumulative_probs <= top_p

    # 至少保留概率最高的 token
    keep[..., 0] = True

    filtered_probs = torch.where(
        keep,
        sorted_probs,
        torch.zeros_like(sorted_probs),
    )

    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_pos = torch.multinomial(filtered_probs, num_samples=1)
    sampled_id = sorted_indices.gather(-1, sampled_pos)

    return sampled_id


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    context_length,
    device,
    temperature,
    top_p,
    stop_at_eot,
):
    ids = tokenizer.encode(prompt)

    if len(ids) == 0:
        raise ValueError("prompt 编码后为空，请换一个 prompt")

    eot_id = None
    if "<|endoftext|>" in tokenizer.special_tokens:
        eot_id = tokenizer.i_vocab.get("<|endoftext|>".encode("utf-8"))

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -context_length:]

        logits = model(idx_cond)
        next_logits = logits[:, -1, :]

        if temperature == 0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)

            if top_p is not None and top_p < 1.0:
                next_id = top_p_sample(probs, top_p)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if stop_at_eot and eot_id is not None and next_id.item() == eot_id:
            break

    out_ids = input_ids[0].tolist()
    return tokenizer.decode(out_ids)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="checkpoints/tinystories_final.pt")
    parser.add_argument("--vocab-path", type=str, default="artifacts/tinystories_vocab.pkl")
    parser.add_argument("--merges-path", type=str, default="artifacts/tinystories_merges.pkl")

    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)

    # 必须和训练时的模型参数一致
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stop-at-eot", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer = load_tokenizer(
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    model = load_model(args, device)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_at_eot=args.stop_at_eot,
    )

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


if __name__ == "__main__":
    main()