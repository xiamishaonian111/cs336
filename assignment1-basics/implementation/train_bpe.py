"""Train BPE tokenizer on TinyStories or OpenWebText dataset."""

import argparse
import json
import time
import psutil
import os
from pathlib import Path

from tokenizer import train_bpe


def get_memory_usage_gb():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "owt"],
        required=True,
        help="Dataset to train on: tinystories or owt (OpenWebText)"
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size (default: 10000 for tinystories, 32000 for owt)"
    )
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Disable multiprocessing"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile profiling"
    )
    args = parser.parse_args()

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Dataset-specific configuration
    if args.dataset == "tinystories":
        input_path = data_dir / f"TinyStoriesV2-GPT4-{args.split}.txt"
        default_vocab_size = 10000
        output_prefix = f"tinystories_{args.split}"
    else:  # owt
        input_path = data_dir / f"owt_{args.split}.txt"
        default_vocab_size = 32000
        output_prefix = f"owt_{args.split}"

    vocab_size = args.vocab_size if args.vocab_size else default_vocab_size
    special_tokens = ["<|endoftext|>"]

    print(f"Training BPE tokenizer on: {input_path}")
    print(f"Dataset: {args.dataset}, Split: {args.split}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"File size: {input_path.stat().st_size / (1024**3):.2f} GB")
    print(f"Multiprocessing: {not args.no_multiprocessing}")
    print()

    # Track memory and time
    initial_memory = get_memory_usage_gb()
    start_time = time.time()

    # Train the tokenizer
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        use_multiprocessing=not args.no_multiprocessing,
    )

    end_time = time.time()
    peak_memory = get_memory_usage_gb()

    # Calculate stats
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    elapsed_hours = elapsed_time / 3600
    memory_used = peak_memory - initial_memory

    print(f"\nTraining completed!")
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
    vocab_path = output_dir / f"{output_prefix}_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab_json, f, indent=2)
    print(f"Saved vocabulary to: {vocab_path}")

    # Serialize merges (as list of hex string pairs)
    merges_json = [[m[0].hex(), m[1].hex()] for m in merges]
    merges_path = output_dir / f"{output_prefix}_merges.json"
    with open(merges_path, "w") as f:
        json.dump(merges_json, f, indent=2)
    print(f"Saved merges to: {merges_path}")

    # Also save a human-readable version of the vocab
    vocab_readable_path = output_dir / f"{output_prefix}_vocab_readable.txt"
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
    import sys

    # Check if profiling is requested
    if "--profile" in sys.argv:
        import cProfile
        import pstats

        # Remove --profile from args before parsing
        sys.argv.remove("--profile")

        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()

        print("\n" + "="*80)
        print("PROFILING RESULTS (top 30 by cumulative time)")
        print("="*80)
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats('cumulative').print_stats(30)

        print("\n" + "="*80)
        print("PROFILING RESULTS (top 30 by total time)")
        print("="*80)
        stats.strip_dirs().sort_stats('tottime').print_stats(30)
    else:
        main()
