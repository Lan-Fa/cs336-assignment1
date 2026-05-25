from __future__ import annotations

import os
from typing import Any

import regex
from cs336_basics import _cpp_tokenizer


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab

        self.i_vocab = {}
        for key in vocab:
            val = vocab[key]
            self.i_vocab[val] = key

        self.merges = merges
        self.special_tokens: list[str] = special_tokens or []

        self.pat = regex.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.cpp_bpe = _cpp_tokenizer.CppBPE(self.i_vocab, self.merges)

    def decode(self, ids):
        return b"".join(self.vocab[token_id] for token_id in ids).decode(
            "utf-8",
            errors="replace",
        )

    def encode_chunk(self, text):
        return self.cpp_bpe.encode_bytes(text.encode("utf-8"))

    def encode_without_special(self, text):
        res = []
        for match in self.pat.finditer(text):
            res.extend(self.encode_chunk(match.group(0)))
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


def run_train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pat = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    cnt = {}

    if len(special_tokens) > 0:
        spt = "|".join(
            regex.escape(spt)
            for spt in sorted(special_tokens, key=len, reverse=True)
        )
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
            cnt[byte_token] = cnt.get(byte_token, 0) + 1

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
                pair_cnt[pair] = pair_cnt.get(pair, 0) + c

        if len(pair_cnt) == 0:
            break

        mx_pr = max(pair_cnt, key=lambda pair: (pair_cnt[pair], pair))
        merges.append(mx_pr)

        new_vocab_token = mx_pr[0] + mx_pr[1]
        vocab[len(vocab)] = new_vocab_token

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
            new_cnt[new_token] = new_cnt.get(new_token, 0) + c

        cnt = new_cnt

    return vocab, merges


def get_tokenizer(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
) -> Any:
    return BPETokenizer(vocab, merges, special_tokens)