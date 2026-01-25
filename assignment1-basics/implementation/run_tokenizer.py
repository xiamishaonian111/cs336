"""Run various tokenizer tasks: compression analysis, throughput estimation, and dataset encoding."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from tokenizer import Tokenizer


def load_tokenizer_from_hex_files(
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """
    Load a tokenizer from vocab and merges files saved in hex format.

    Args:
        vocab_filepath: Path to vocab JSON (token_id -> hex string)
        merges_filepath: Path to merges JSON (list of [hex1, hex2])
        special_tokens: Optional list of special tokens

    Returns:
        Tokenizer instance
    """
    # Load vocab: {token_id_str: hex_string} -> {int: bytes}
    with open(vocab_filepath, "r") as f:
        vocab_data = json.load(f)
    vocab = {int(k): bytes.fromhex(v) for k, v in vocab_data.items()}

    # Load merges: [[hex1, hex2], ...] -> [(bytes, bytes), ...]
    with open(merges_filepath, "r") as f:
        merges_data = json.load(f)
    merges = [(bytes.fromhex(pair[0]), bytes.fromhex(pair[1])) for pair in merges_data]

    return Tokenizer(vocab, merges, special_tokens)


def sample_documents(filepath: str, num_docs: int, separator: str = "<|endoftext|>", seed: int = 42) -> list[str]:
    """
    Sample random documents from a text file.

    Args:
        filepath: Path to the text file
        num_docs: Number of documents to sample
        separator: Document separator string
        seed: Random seed for reproducibility

    Returns:
        List of sampled document strings
    """
    random.seed(seed)

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into documents
    documents = content.split(separator)
    # Remove empty documents and strip whitespace
    documents = [doc.strip() for doc in documents if doc.strip()]

    # Sample documents
    if len(documents) <= num_docs:
        return documents

    return random.sample(documents, num_docs)


def calculate_compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> dict:
    """
    Calculate compression ratio for a list of documents.

    Args:
        tokenizer: Tokenizer to use for encoding
        documents: List of document strings

    Returns:
        Dictionary with compression statistics
    """
    total_bytes = 0
    total_tokens = 0
    per_doc_ratios = []

    for doc in documents:
        # Calculate bytes (UTF-8 encoding)
        doc_bytes = len(doc.encode("utf-8"))

        # Encode to tokens
        token_ids = tokenizer.encode(doc)
        doc_tokens = len(token_ids)

        total_bytes += doc_bytes
        total_tokens += doc_tokens

        if doc_tokens > 0:
            per_doc_ratios.append(doc_bytes / doc_tokens)

    return {
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "compression_ratio": total_bytes / total_tokens if total_tokens > 0 else 0,
        "per_doc_ratios": per_doc_ratios,
        "avg_doc_ratio": sum(per_doc_ratios) / len(per_doc_ratios) if per_doc_ratios else 0,
    }


def experiment1_compression_ratio(args, data_dir: Path, output_dir: Path):
    """
    Experiment 1: Sample 10 documents from TinyStories and OpenWebText.
    Calculate compression ratio (bytes/token) for each tokenizer.
    """
    print("=" * 70)
    print("Experiment 1: Compression Ratio Analysis")
    print("=" * 70)

    # Dataset configurations
    datasets = {
        "TinyStories": {
            "data_path": data_dir / "TinyStoriesV2-GPT4-train.txt",
            "vocab_path": output_dir / "tinystories_vocab.json",
            "merges_path": output_dir / "tinystories_merges.json",
            "vocab_size": 10000,
        },
        "OpenWebText": {
            "data_path": data_dir / "owt_train.txt",
            "vocab_path": output_dir / "owt_train_vocab.json",
            "merges_path": output_dir / "owt_train_merges.json",
            "vocab_size": 32000,
        },
    }

    special_tokens = ["<|endoftext|>"]

    print(f"Sampling {args.num_docs} documents from each dataset (seed={args.seed})")

    for dataset_name, config in datasets.items():
        print(f"\n{dataset_name} (vocab size: {config['vocab_size']})")
        print("-" * 50)

        # Load tokenizer
        tokenizer = load_tokenizer_from_hex_files(
            str(config["vocab_path"]),
            str(config["merges_path"]),
            special_tokens,
        )
        print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")

        # Sample documents
        documents = sample_documents(
            str(config["data_path"]),
            args.num_docs,
            separator="<|endoftext|>",
            seed=args.seed,
        )
        print(f"Sampled {len(documents)} documents")

        # Calculate compression ratio
        stats = calculate_compression_ratio(tokenizer, documents)

        print(f"\nCompression Statistics:")
        print(f"  Total bytes:        {stats['total_bytes']:,}")
        print(f"  Total tokens:       {stats['total_tokens']:,}")
        print(f"  Compression ratio:  {stats['compression_ratio']:.4f} bytes/token")
        print(f"  Avg per-doc ratio:  {stats['avg_doc_ratio']:.4f} bytes/token")

        print(f"\nPer-document ratios:")
        for i, ratio in enumerate(stats["per_doc_ratios"]):
            print(f"  Doc {i+1}: {ratio:.4f} bytes/token")


def experiment2_cross_tokenizer(args, data_dir: Path, output_dir: Path):
    """
    Experiment 2: Tokenize OpenWebText sample with TinyStories tokenizer.
    Compare compression ratios.
    """
    print("=" * 70)
    print("Experiment 2: Cross-Tokenizer Analysis (OWT with TinyStories tokenizer)")
    print("=" * 70)

    special_tokens = ["<|endoftext|>"]

    # Load both tokenizers
    tinystories_tokenizer = load_tokenizer_from_hex_files(
        str(output_dir / "tinystories_vocab.json"),
        str(output_dir / "tinystories_merges.json"),
        special_tokens,
    )
    owt_tokenizer = load_tokenizer_from_hex_files(
        str(output_dir / "owt_train_vocab.json"),
        str(output_dir / "owt_train_merges.json"),
        special_tokens,
    )

    # Sample OpenWebText documents
    owt_documents = sample_documents(
        str(data_dir / "owt_train.txt"),
        args.num_docs,
        separator="<|endoftext|>",
        seed=args.seed,
    )
    print(f"Sampled {len(owt_documents)} OpenWebText documents")

    # Compare compression ratios
    print("\nOpenWebText with OpenWebText tokenizer (32K vocab):")
    print("-" * 50)
    owt_stats = calculate_compression_ratio(owt_tokenizer, owt_documents)
    print(f"  Compression ratio: {owt_stats['compression_ratio']:.4f} bytes/token")
    print(f"  Total tokens: {owt_stats['total_tokens']:,}")

    print("\nOpenWebText with TinyStories tokenizer (10K vocab):")
    print("-" * 50)
    ts_stats = calculate_compression_ratio(tinystories_tokenizer, owt_documents)
    print(f"  Compression ratio: {ts_stats['compression_ratio']:.4f} bytes/token")
    print(f"  Total tokens: {ts_stats['total_tokens']:,}")

    print("\nComparison:")
    print("-" * 50)
    ratio_diff = ts_stats['compression_ratio'] / owt_stats['compression_ratio']
    token_diff = ts_stats['total_tokens'] / owt_stats['total_tokens']
    print(f"  TinyStories tokenizer produces {token_diff:.2f}x more tokens")
    print(f"  Compression ratio is {ratio_diff:.2f}x {'worse' if ratio_diff < 1 else 'better'}")

    # Qualitative analysis: show example tokenization
    print("\nQualitative Example (first 200 chars of first document):")
    print("-" * 50)
    sample_text = owt_documents[0][:200]
    print(f"Text: {repr(sample_text)}")

    owt_tokens = owt_tokenizer.encode(sample_text)
    ts_tokens = tinystories_tokenizer.encode(sample_text)

    print(f"\nOWT tokenizer ({len(owt_tokens)} tokens):")
    owt_decoded = [owt_tokenizer.decode([t]) for t in owt_tokens]
    print(f"  {owt_decoded}")

    print(f"\nTinyStories tokenizer ({len(ts_tokens)} tokens):")
    ts_decoded = [tinystories_tokenizer.decode([t]) for t in ts_tokens]
    print(f"  {ts_decoded}")


def experiment3_throughput(args, data_dir: Path, output_dir: Path):
    """
    Experiment 3: Estimate tokenizer throughput (bytes/second).
    Estimate time to tokenize the Pile dataset (825GB).
    """
    print("=" * 70)
    print("Experiment 3: Throughput Estimation")
    print("=" * 70)

    special_tokens = ["<|endoftext|>"]

    # Load tokenizers
    tokenizers = {
        "TinyStories (10K)": load_tokenizer_from_hex_files(
            str(output_dir / "tinystories_vocab.json"),
            str(output_dir / "tinystories_merges.json"),
            special_tokens,
        ),
        "OpenWebText (32K)": load_tokenizer_from_hex_files(
            str(output_dir / "owt_train_vocab.json"),
            str(output_dir / "owt_train_merges.json"),
            special_tokens,
        ),
    }

    # Load sample data for benchmarking
    sample_size_mb = args.throughput_sample_mb
    print(f"Loading {sample_size_mb} MB sample for throughput estimation...")

    with open(data_dir / "owt_train.txt", "r", encoding="utf-8") as f:
        # Read approximately sample_size_mb of text
        sample_text = f.read(sample_size_mb * 1024 * 1024)

    actual_bytes = len(sample_text.encode("utf-8"))
    print(f"Loaded {actual_bytes / (1024*1024):.2f} MB of text")

    pile_size_gb = 825

    for name, tokenizer in tokenizers.items():
        print(f"\n{name}:")
        print("-" * 50)

        # Warm up
        _ = tokenizer.encode(sample_text[:1000])

        # Benchmark
        start_time = time.time()
        tokens = tokenizer.encode(sample_text)
        elapsed = time.time() - start_time

        throughput_bytes_per_sec = actual_bytes / elapsed
        throughput_mb_per_sec = throughput_bytes_per_sec / (1024 * 1024)

        # Estimate time for Pile
        pile_bytes = pile_size_gb * 1024 * 1024 * 1024
        pile_time_seconds = pile_bytes / throughput_bytes_per_sec
        pile_time_hours = pile_time_seconds / 3600
        pile_time_days = pile_time_hours / 24

        print(f"  Tokens produced: {len(tokens):,}")
        print(f"  Time elapsed: {elapsed:.2f} seconds")
        print(f"  Throughput: {throughput_bytes_per_sec:,.0f} bytes/second")
        print(f"  Throughput: {throughput_mb_per_sec:.2f} MB/second")
        print(f"\n  Estimated time for Pile ({pile_size_gb} GB):")
        print(f"    {pile_time_seconds:,.0f} seconds")
        print(f"    {pile_time_hours:.2f} hours")
        print(f"    {pile_time_days:.2f} days")


def experiment4_encode_datasets(args, data_dir: Path, output_dir: Path):
    """
    Experiment 4: Encode training and validation datasets into NumPy arrays (uint16).
    """
    print("=" * 70)
    print("Experiment 4: Encode Datasets to NumPy Arrays")
    print("=" * 70)

    special_tokens = ["<|endoftext|>"]

    # Dataset configurations
    datasets = [
        {
            "name": "TinyStories",
            "tokenizer_vocab": output_dir / "tinystories_vocab.json",
            "tokenizer_merges": output_dir / "tinystories_merges.json",
            "splits": {
                "train": data_dir / "TinyStoriesV2-GPT4-train.txt",
                "valid": data_dir / "TinyStoriesV2-GPT4-valid.txt",
            },
        },
        {
            "name": "OpenWebText",
            "tokenizer_vocab": output_dir / "owt_train_vocab.json",
            "tokenizer_merges": output_dir / "owt_train_merges.json",
            "splits": {
                "train": data_dir / "owt_train.txt",
                "valid": data_dir / "owt_valid.txt",
            },
        },
    ]

    print("Why uint16?")
    print("-" * 50)
    print("  - TinyStories vocab size: 10,000 (max value fits in uint16: 0-65535)")
    print("  - OpenWebText vocab size: 32,000 (max value fits in uint16: 0-65535)")
    print("  - uint16 uses 2 bytes per token (vs 4 bytes for uint32 or 8 for int64)")
    print("  - This halves memory usage compared to int32, important for large datasets")
    print()

    for dataset_config in datasets:
        name = dataset_config["name"]
        print(f"\n{name}")
        print("=" * 50)

        # Load tokenizer
        tokenizer = load_tokenizer_from_hex_files(
            str(dataset_config["tokenizer_vocab"]),
            str(dataset_config["tokenizer_merges"]),
            special_tokens,
        )
        print(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens")

        # Verify vocab size fits in uint16
        max_token_id = max(tokenizer.vocab.keys())
        assert max_token_id < 65536, f"Vocab too large for uint16: {max_token_id}"
        print(f"Max token ID: {max_token_id} (fits in uint16)")

        for split_name, data_path in dataset_config["splits"].items():
            print(f"\n  {split_name} split:")
            print(f"  {'-' * 40}")

            if not data_path.exists():
                print(f"  WARNING: {data_path} not found, skipping")
                continue

            # Get file size
            file_size_gb = data_path.stat().st_size / (1024**3)
            print(f"  Input file: {data_path.name} ({file_size_gb:.2f} GB)")

            # Encode using encode_iterable for memory efficiency
            print(f"  Encoding...")
            start_time = time.time()

            token_ids = []
            with open(data_path, "r", encoding="utf-8") as f:
                for token_id in tokenizer.encode_iterable(f):
                    token_ids.append(token_id)

            elapsed = time.time() - start_time
            print(f"  Encoding time: {elapsed:.2f} seconds")

            # Convert to NumPy array
            token_array = np.array(token_ids, dtype=np.uint16)
            print(f"  Total tokens: {len(token_array):,}")
            print(f"  Array size: {token_array.nbytes / (1024**2):.2f} MB")

            # Save to file
            output_path = output_dir / f"{name.lower()}_{split_name}_tokens.npy"
            np.save(output_path, token_array)
            print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run tokenizer experiments")
    parser.add_argument(
        "experiment",
        choices=["experiment1", "experiment2", "experiment3", "experiment4", "all"],
        help="Which experiment to run"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=10,
        help="Number of documents to sample (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--throughput-sample-mb",
        type=int,
        default=10,
        help="Sample size in MB for throughput estimation (default: 10)"
    )
    args = parser.parse_args()

    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if args.experiment == "experiment1" or args.experiment == "all":
        experiment1_compression_ratio(args, data_dir, output_dir)

    if args.experiment == "experiment2" or args.experiment == "all":
        experiment2_cross_tokenizer(args, data_dir, output_dir)

    if args.experiment == "experiment3" or args.experiment == "all":
        experiment3_throughput(args, data_dir, output_dir)

    if args.experiment == "experiment4" or args.experiment == "all":
        experiment4_encode_datasets(args, data_dir, output_dir)

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
