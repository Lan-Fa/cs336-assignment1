import pickle
import time
from pathlib import Path

import numpy as np
import regex

from cs336_basics.tokenizer import run_train_bpe, get_tokenizer

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VALID_PATH = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"

# 用于训练 BPE 的小样本，不直接用全量 TinyStories
BPE_SAMPLE_PATH = DATA_DIR / "tinystories_sample_5M.txt"
BPE_SAMPLE_BYTES = 5_000_000

VOCAB_PATH = ARTIFACTS_DIR / "tinystories_vocab.pkl"
MERGES_PATH = ARTIFACTS_DIR / "tinystories_merges.pkl"

TRAIN_IDS_PATH = DATA_DIR / "tinystories_train_ids.npy"
VALID_IDS_PATH = DATA_DIR / "tinystories_valid_ids.npy"

VOCAB_SIZE = 1000
SPECIAL_TOKENS = ["<|endoftext|>"]

# 如果你想强制重新训练 tokenizer，把这里改成 True
FORCE_RETRAIN_TOKENIZER = False


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds / 60:.2f}min"
    return f"{seconds / 3600:.2f}h"

def format_rate(count: int | float, seconds: float, unit: str) -> str:
    if seconds <= 0:
        return f"inf {unit}/s"
    return f"{count / seconds:.2f} {unit}/s"


def encode_with_timing(tokenizer, text: str, name: str):
    print(f"[{name}] Starting detailed encode timing...")

    total_start = time.perf_counter()

    split_time = 0.0
    regex_next_time = 0.0
    group_time = 0.0
    cpp_encode_time = 0.0
    extend_time = 0.0
    special_time = 0.0

    num_parts = 0
    num_special_parts = 0
    num_regex_chunks = 0
    num_chars_encoded = 0
    num_ids = 0

    res = []

    # 1. special token split
    split_start = time.perf_counter()

    if not tokenizer.special_tokens:
        parts = [text]
        sp_token = set()
    else:
        special_tokens = sorted(tokenizer.special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(regex.escape(tok) for tok in special_tokens) + ")"
        parts = regex.split(special_pattern, text)
        sp_token = set(tokenizer.special_tokens)

    split_time = time.perf_counter() - split_start

    print(f"[{name}] Special split done in {format_time(split_time)}")
    print(f"[{name}] Number of split parts: {len(parts)}")

    progress_last_print = time.perf_counter()
    progress_start = time.perf_counter()

    # 2. encode parts
    for part in parts:
        if part == "":
            continue

        num_parts += 1

        if part in sp_token:
            t0 = time.perf_counter()
            res.append(tokenizer.i_vocab[part.encode("utf-8")])
            t1 = time.perf_counter()

            special_time += t1 - t0
            num_special_parts += 1
            num_ids += 1
            continue

        # 手动 next(iterator)，这样能统计 regex 找下一个 match 的时间
        it = tokenizer.pat.finditer(part)

        while True:
            t0 = time.perf_counter()
            try:
                match = next(it)
            except StopIteration:
                regex_next_time += time.perf_counter() - t0
                break
            t1 = time.perf_counter()
            regex_next_time += t1 - t0

            t0 = time.perf_counter()
            chunk = match.group(0)
            t1 = time.perf_counter()
            group_time += t1 - t0

            t0 = time.perf_counter()
            ids = tokenizer.encode_chunk(chunk)
            t1 = time.perf_counter()
            cpp_encode_time += t1 - t0

            t0 = time.perf_counter()
            res.extend(ids)
            t1 = time.perf_counter()
            extend_time += t1 - t0

            num_regex_chunks += 1
            num_chars_encoded += len(chunk)
            num_ids += len(ids)

            now = time.perf_counter()

            if now - progress_last_print >= 10:
                elapsed = now - progress_start
                print(
                    f"[{name}] progress: "
                    f"chunks={num_regex_chunks:,}, "
                    f"chars={num_chars_encoded:,}, "
                    f"ids={num_ids:,}, "
                    f"elapsed={format_time(elapsed)}, "
                    f"char_rate={format_rate(num_chars_encoded, elapsed, 'chars')}, "
                    f"id_rate={format_rate(num_ids, elapsed, 'ids')}"
                )
                print(
                    f"[{name}] timing so far: "
                    f"regex_next={format_time(regex_next_time)}, "
                    f"group={format_time(group_time)}, "
                    f"cpp_encode={format_time(cpp_encode_time)}, "
                    f"extend={format_time(extend_time)}"
                )
                progress_last_print = now

    total_time = time.perf_counter() - total_start

    print(f"[{name}] Detailed encode finished.")
    print(f"[{name}] parts: {num_parts:,}")
    print(f"[{name}] special parts: {num_special_parts:,}")
    print(f"[{name}] regex chunks: {num_regex_chunks:,}")
    print(f"[{name}] chars encoded: {num_chars_encoded:,}")
    print(f"[{name}] ids produced: {num_ids:,}")

    print(f"[{name}] split time: {format_time(split_time)}")
    print(f"[{name}] regex next time: {format_time(regex_next_time)}")
    print(f"[{name}] match.group time: {format_time(group_time)}")
    print(f"[{name}] C++ encode_chunk time: {format_time(cpp_encode_time)}")
    print(f"[{name}] list extend time: {format_time(extend_time)}")
    print(f"[{name}] special token time: {format_time(special_time)}")
    print(f"[{name}] total encode_with_timing time: {format_time(total_time)}")

    print(f"[{name}] char rate: {format_rate(num_chars_encoded, total_time, 'chars')}")
    print(f"[{name}] id rate: {format_rate(num_ids, total_time, 'ids')}")

    return res

def create_bpe_sample():
    if BPE_SAMPLE_PATH.exists():
        print(f"BPE sample already exists: {BPE_SAMPLE_PATH}")
        return

    print(f"Creating BPE sample: {BPE_SAMPLE_PATH}")

    start = time.perf_counter()

    with open(TRAIN_PATH, "rb") as src:
        data = src.read(BPE_SAMPLE_BYTES)

    with open(BPE_SAMPLE_PATH, "wb") as dst:
        dst.write(data)

    elapsed = time.perf_counter() - start
    print(f"Created BPE sample in {format_time(elapsed)}")


def load_or_train_tokenizer():
    if (
            VOCAB_PATH.exists()
            and MERGES_PATH.exists()
            and not FORCE_RETRAIN_TOKENIZER
    ):
        print("Loading existing vocab and merges...")

        start = time.perf_counter()

        with open(VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)

        with open(MERGES_PATH, "rb") as f:
            merges = pickle.load(f)

        elapsed = time.perf_counter() - start
        print(f"Loaded vocab and merges in {format_time(elapsed)}")

        return vocab, merges

    print("Training BPE tokenizer...")

    start = time.perf_counter()

    vocab, merges = run_train_bpe(
        input_path=BPE_SAMPLE_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    elapsed = time.perf_counter() - start
    print(f"Trained BPE tokenizer in {format_time(elapsed)}")

    print("Saving vocab and merges...")

    start = time.perf_counter()

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    with open(MERGES_PATH, "wb") as f:
        pickle.dump(merges, f)

    elapsed = time.perf_counter() - start
    print(f"Saved vocab and merges in {format_time(elapsed)}")

    return vocab, merges


def encode_and_save(tokenizer, input_path: Path, output_path: Path, name: str):
    print(f"Encoding {name} data in streaming mode...")

    if output_path.exists():
        print(f"{output_path} already exists, skipping {name} encoding.")
        return np.load(output_path, mmap_mode="r")

    total_start = time.perf_counter()

    file_size = input_path.stat().st_size
    print(f"{name} file size: {file_size / 1024 / 1024:.2f} MB")

    dtype = np.uint16 if VOCAB_SIZE <= 65535 else np.uint32

    # Pass 1: count total token ids
    print(f"[{name}] Pass 1/2: counting tokens...")

    count_start = time.perf_counter()

    total_ids = 0
    total_chars = 0
    total_lines = 0
    last_print = time.perf_counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            ids = tokenizer.encode(line)

            total_ids += len(ids)
            total_chars += len(line)
            total_lines += 1

            now = time.perf_counter()
            if now - last_print >= 10:
                elapsed = now - count_start
                print(
                    f"[{name}] count progress: "
                    f"lines={total_lines:,}, "
                    f"chars={total_chars:,}, "
                    f"ids={total_ids:,}, "
                    f"elapsed={format_time(elapsed)}, "
                    f"char_rate={total_chars / elapsed:.2f} chars/s, "
                    f"id_rate={total_ids / elapsed:.2f} ids/s"
                )
                last_print = now

    count_elapsed = time.perf_counter() - count_start

    print(f"[{name}] token count finished in {format_time(count_elapsed)}")
    print(f"[{name}] total lines: {total_lines:,}")
    print(f"[{name}] total chars: {total_chars:,}")
    print(f"[{name}] total ids: {total_ids:,}")

    # Pass 2: write token ids directly into npy memmap
    print(f"[{name}] Pass 2/2: writing ids to {output_path}...")

    write_start = time.perf_counter()

    arr = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=(total_ids,),
    )

    pos = 0
    total_chars = 0
    total_lines = 0
    last_print = time.perf_counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            ids = tokenizer.encode(line)
            n = len(ids)

            arr[pos:pos + n] = np.asarray(ids, dtype=dtype)
            pos += n

            total_chars += len(line)
            total_lines += 1

            now = time.perf_counter()
            if now - last_print >= 10:
                elapsed = now - write_start
                print(
                    f"[{name}] write progress: "
                    f"lines={total_lines:,}, "
                    f"chars={total_chars:,}, "
                    f"written_ids={pos:,}/{total_ids:,}, "
                    f"elapsed={format_time(elapsed)}, "
                    f"id_rate={pos / elapsed:.2f} ids/s"
                )
                last_print = now

    arr.flush()

    write_elapsed = time.perf_counter() - write_start
    total_elapsed = time.perf_counter() - total_start

    print(f"Saved {name} ids to {output_path}")
    print(f"{name} ids shape: {arr.shape}")
    print(f"{name} ids dtype: {arr.dtype}")
    print(f"{name} ids memory: {arr.nbytes / 1024 / 1024:.2f} MB")
    print(f"[{name}] count time: {format_time(count_elapsed)}")
    print(f"[{name}] write time: {format_time(write_elapsed)}")
    print(f"Total {name} processing time: {format_time(total_elapsed)}")

    return arr


def main():
    total_start = time.perf_counter()

    print("Preparing TinyStories data...")
    print(f"VOCAB_SIZE = {VOCAB_SIZE}")
    print(f"SPECIAL_TOKENS = {SPECIAL_TOKENS}")

    create_bpe_sample()

    vocab, merges = load_or_train_tokenizer()

    tokenizer = get_tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=SPECIAL_TOKENS,
    )

    encode_and_save(
        tokenizer=tokenizer,
        input_path=TRAIN_PATH,
        output_path=TRAIN_IDS_PATH,
        name="train",
    )

    encode_and_save(
        tokenizer=tokenizer,
        input_path=VALID_PATH,
        output_path=VALID_IDS_PATH,
        name="valid",
    )

    total_elapsed = time.perf_counter() - total_start

    print("Done.")
    print(f"vocab saved to: {VOCAB_PATH}")
    print(f"merges saved to: {MERGES_PATH}")
    print(f"train ids saved to: {TRAIN_IDS_PATH}")
    print(f"valid ids saved to: {VALID_IDS_PATH}")
    print(f"Total prepare time: {format_time(total_elapsed)}")


if __name__ == "__main__":
    main()
