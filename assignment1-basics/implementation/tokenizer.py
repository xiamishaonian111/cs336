"""BPE (Byte Pair Encoding) tokenizer training implementation."""

from __future__ import annotations

import os
import re
import regex
import multiprocessing
import time
from functools import partial

from cs336_basics.pretokenization_example import find_chunk_boundaries


# Pre-tokenization regex pattern (GPT-2 style) - pre-compiled for performance
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = regex.compile(PAT)


def _pretokenize_chunk(
    chunk_range: tuple[int, int],
    input_path: str | os.PathLike,
    special_token_pattern: str,
) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize a chunk of the input file.

    Args:
        chunk_range: (start, end) byte offsets in the file
        input_path: Path to the input file
        special_token_pattern: Regex pattern to split on special tokens

    Returns:
        Dictionary mapping token tuples to their counts in this chunk
    """
    start, end = chunk_range

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk = chunk_bytes.decode("utf-8", errors="ignore")

    # Split on special tokens
    if special_token_pattern:
        chunks = re.split(special_token_pattern, chunk)
    else:
        chunks = [chunk]

    # Pre-tokenize and count frequencies
    # Use raw bytes as key first (much faster), then convert to tuple format at the end
    pre_tokenization_raw: dict[bytes, int] = {}
    for text_chunk in chunks:
        for match in PAT_COMPILED.finditer(text_chunk):
            token_bytes = match.group().encode("utf-8")
            pre_tokenization_raw[token_bytes] = pre_tokenization_raw.get(token_bytes, 0) + 1

    # Convert to tuple format for BPE processing
    pre_tokenization: dict[tuple[bytes, ...], int] = {}
    for token_bytes, count in pre_tokenization_raw.items():
        token_tuple = tuple(bytes([b]) for b in token_bytes)
        pre_tokenization[token_tuple] = count

    return pre_tokenization


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    use_multiprocessing: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size
                   (including the initial byte vocabulary, vocabulary items produced
                   from merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. These special
                       tokens do not otherwise affect BPE training.

    Returns:
        vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
              to bytes (token bytes).
        merges: A list of BPE merges produced from training. Each list item is a tuple
               of bytes (<token1>, <token2>), representing that <token1> was merged with
               <token2>. The merges should be ordered by order of creation.
    """
    # Build special token pattern for splitting
    special_token_pattern = "|".join(re.escape(token) for token in special_tokens) if special_tokens else ""

    t0 = time.time()
    if use_multiprocessing:
        # Determine the split token for chunking (use first special token or default)
        split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"

        # Find chunk boundaries for parallel processing
        num_processes = multiprocessing.cpu_count()
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

        # Create chunk ranges for parallel processing
        chunk_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        # Parallel pre-tokenization
        worker_fn = partial(_pretokenize_chunk, input_path=input_path, special_token_pattern=special_token_pattern)

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(worker_fn, chunk_ranges)

        t1 = time.time()
        print(f"[TIMING] Pool.map completed: {t1 - t0:.2f}s")

        # Merge results from all processes
        pre_tokenization: dict[tuple[bytes, ...], int] = {}
        for result in results:
            for token_tuple, count in result.items():
                pre_tokenization[token_tuple] = pre_tokenization.get(token_tuple, 0) + count
        t2 = time.time()
        print(f"[TIMING] Merge results: {t2 - t1:.2f}s")
        print(f"[TIMING] Total pre-tokenization: {t2 - t0:.2f}s")
        print(f"[TIMING] Unique tokens: {len(pre_tokenization)}")
    else:
        # Single-process pre-tokenization
        with open(input_path, "rb") as f:
            file_size = f.seek(0, os.SEEK_END)
        pre_tokenization = _pretokenize_chunk((0, file_size), input_path, special_token_pattern)

    # TODO 4: Initialize vocabulary with base byte tokens (0-255)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # TODO 5: Add special tokens to vocabulary
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    # TODO 6: Perform BPE training from pre-tokenization result - iteratively merge most frequent pairs
    # Optimized version: build pair_freq once, then incrementally update after each merge
    # 1. Only pairs adjacent to merge positions change
    # 2. Compute old_pairs and new_pairs for affected tokens, update pair_freq with the diff

    # TODO 6.1: Build initial pair frequency counts and reverse index (once, before the loop)
    t3 = time.time()
    pair_freq: dict[tuple[bytes, bytes], int] = {}
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for token, freq in pre_tokenization.items():
        for j in range(len(token) - 1):
            pair = (token[j], token[j + 1])
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
            if pair not in pair_to_tokens:
                pair_to_tokens[pair] = set()
            pair_to_tokens[pair].add(token)
    t4 = time.time()
    print(f"[TIMING] Build pair_freq: {t4 - t3:.2f}s")
    print(f"[TIMING] Unique pairs: {len(pair_freq)}")

    # Initialize the list to store merges
    merges: list[tuple[bytes, bytes]] = []

    # Repeat the merging process
    num_merges = vocab_size - 256 - len(special_tokens)
    for merge_idx in range(num_merges):
        # TODO 6.2: find the most frequent pair (with lexicographic tie-breaking)
        if not pair_freq:
            break  # No more pairs to merge

        max_freq_pair: tuple[bytes, bytes] | None = None
        max_freq = 0
        for pair, freq in pair_freq.items():
            if freq > max_freq or (freq == max_freq and pair > max_freq_pair):
                max_freq = freq
                max_freq_pair = pair

        # TODO 6.3: add the merge to merges
        merges.append(max_freq_pair)

        # Log progress every 1,000 merges
        if (merge_idx + 1) % 1000 == 0:
            print(f"Merge {merge_idx + 1}/{num_merges} completed")

        # TODO 6.4: add the merged pair to vocab
        merged_token = max_freq_pair[0] + max_freq_pair[1]
        vocab[len(vocab)] = merged_token

        # TODO 6.5: update pre-tokenization, pair_freq, and pair_to_tokens incrementally
        # Use reverse index to only iterate over tokens that contain the merged pair
        tokens_to_delete: list[tuple[bytes, ...]] = []
        tokens_to_add: dict[tuple[bytes, ...], int] = {}
        pairs_to_check: set[tuple[bytes, bytes]] = set()  # Track pairs that might become zero

        # Only iterate over tokens that contain the merged pair
        affected_tokens = list(pair_to_tokens.get(max_freq_pair, set()))
        for token in affected_tokens:
            freq = pre_tokenization[token]

            # Compute old pairs for this token
            old_pairs: dict[tuple[bytes, bytes], int] = {}
            for k in range(len(token) - 1):
                pair = (token[k], token[k + 1])
                old_pairs[pair] = old_pairs.get(pair, 0) + 1

            # Build new token by merging
            new_token: list[bytes] = []
            j = 0
            while j < len(token):
                if j < len(token) - 1 and (token[j], token[j + 1]) == max_freq_pair:
                    new_token.append(merged_token)
                    j += 2
                else:
                    new_token.append(token[j])
                    j += 1
            new_key = tuple(new_token)

            # Compute new pairs for the merged token
            new_pairs: dict[tuple[bytes, bytes], int] = {}
            for k in range(len(new_key) - 1):
                pair = (new_key[k], new_key[k + 1])
                new_pairs[pair] = new_pairs.get(pair, 0) + 1

            # Update pair_freq: subtract old pairs, add new pairs
            for pair, count in old_pairs.items():
                pair_freq[pair] -= count * freq
                pairs_to_check.add(pair)
            for pair, count in new_pairs.items():
                pair_freq[pair] = pair_freq.get(pair, 0) + count * freq

            # Update pair_to_tokens: remove old token from old pairs, add new token to new pairs
            for pair in old_pairs:
                pair_to_tokens[pair].discard(token)
            for pair in new_pairs:
                if pair not in pair_to_tokens:
                    pair_to_tokens[pair] = set()
                pair_to_tokens[pair].add(new_key)

            # Mark for update in pre_tokenization
            tokens_to_delete.append(token)
            tokens_to_add[new_key] = tokens_to_add.get(new_key, 0) + freq

        # Apply changes to pre_tokenization
        for token in tokens_to_delete:
            del pre_tokenization[token]
        for token, freq in tokens_to_add.items():
            pre_tokenization[token] = pre_tokenization.get(token, 0) + freq

        # Clean up zero or negative counts from pair_freq (in-place)
        for pair in pairs_to_check:
            if pair_freq.get(pair, 0) <= 0:
                pair_freq.pop(pair, None)


    # TODO: Return the final vocabulary and merges list
    return vocab, merges
