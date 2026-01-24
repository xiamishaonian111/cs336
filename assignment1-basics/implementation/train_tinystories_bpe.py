"""Train BPE tokenizer on TinyStories dataset."""

import json
import time
import psutil
import os
import cProfile
import pstats
from pathlib import Path

from tokenizer import train_bpe


def get_memory_usage_gb():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    input_path = data_dir / "TinyStoriesV2-GPT4-train.txt"  # Full training dataset
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Training parameters
    vocab_size = 10000  # Minimal merges to isolate pre-tokenization time
    special_tokens = ["<|endoftext|>"]

    print(f"Training BPE tokenizer on: {input_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"File size: {input_path.stat().st_size / (1024**3):.2f} GB")
    print()

    # Track memory and time
    initial_memory = get_memory_usage_gb()
    start_time = time.time()

    # Train the tokenizer
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        use_multiprocessing=True,
    )

    end_time = time.time()
    peak_memory = get_memory_usage_gb()

    # Calculate stats
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    elapsed_hours = elapsed_time / 3600
    memory_used = peak_memory - initial_memory

    print(f"Training completed!")
    print(f"Time: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes, {elapsed_hours:.4f} hours)")
    print(f"Memory usage: {peak_memory:.2f} GB (delta: {memory_used:.2f} GB)")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print()

    # Find the longest token
    longest_token = max(vocab.values(), key=len)
    longest_token_id = [k for k, v in vocab.items() if v == longest_token][0]

    print(f"Longest token:")
    print(f"  ID: {longest_token_id}")
    print(f"  Bytes: {longest_token}")
    print(f"  Length: {len(longest_token)} bytes")
    try:
        decoded = longest_token.decode("utf-8")
        print(f"  Decoded: '{decoded}'")
    except UnicodeDecodeError:
        print(f"  Decoded: (cannot decode as UTF-8)")
    print()

    # Serialize vocab to JSON (convert bytes to hex strings for JSON compatibility)
    vocab_json = {str(k): v.hex() for k, v in vocab.items()}
    vocab_path = output_dir / "tinystories_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab_json, f, indent=2)
    print(f"Saved vocabulary to: {vocab_path}")

    # Serialize merges (as list of hex string pairs)
    merges_json = [[m[0].hex(), m[1].hex()] for m in merges]
    merges_path = output_dir / "tinystories_merges.json"
    with open(merges_path, "w") as f:
        json.dump(merges_json, f, indent=2)
    print(f"Saved merges to: {merges_path}")

    # Also save a human-readable version of the vocab
    vocab_readable_path = output_dir / "tinystories_vocab_readable.txt"
    with open(vocab_readable_path, "w", encoding="utf-8") as f:
        for token_id in sorted(vocab.keys()):
            token_bytes = vocab[token_id]
            try:
                decoded = token_bytes.decode("utf-8")
                f.write(f"{token_id}\t{repr(decoded)}\n")
            except UnicodeDecodeError:
                f.write(f"{token_id}\t{token_bytes.hex()}\n")
    print(f"Saved readable vocabulary to: {vocab_readable_path}")


if __name__ == "__main__":
    # Profile the training
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    # Print profiling results
    print("\n" + "="*80)
    print("PROFILING RESULTS (top 30 by cumulative time)")
    print("="*80)
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(30)

    print("\n" + "="*80)
    print("PROFILING RESULTS (top 30 by total time)")
    print("="*80)
    stats.strip_dirs().sort_stats('tottime').print_stats(30)
