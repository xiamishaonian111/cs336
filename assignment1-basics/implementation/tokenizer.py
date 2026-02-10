"""BPE (Byte Pair Encoding) tokenizer training and encoding/decoding implementation."""

from __future__ import annotations

import json
import math
import os
import re
import regex
import multiprocessing
import time
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any

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
    num_workers: int | None = None,
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
        num_processes = num_workers if num_workers else multiprocessing.cpu_count()

        # Calculate number of chunks based on max chunk size (256 MB), not worker count
        # Workers will process chunks in streaming fashion via imap_unordered
        max_chunk_size = 256 * 1024 * 1024  # 256 MB
        with open(input_path, "rb") as f:
            file_size = f.seek(0, os.SEEK_END)
        num_chunks = max(1, math.ceil(file_size / max_chunk_size))

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)

        # Create chunk ranges for parallel processing
        chunk_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        # Parallel pre-tokenization
        worker_fn = partial(_pretokenize_chunk, input_path=input_path, special_token_pattern=special_token_pattern)

        # Incrementally merge results as each worker completes (memory-efficient)
        pre_tokenization: dict[tuple[bytes, ...], int] = {}
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(worker_fn, chunk_ranges):
                for token_tuple, count in result.items():
                    pre_tokenization[token_tuple] = pre_tokenization.get(token_tuple, 0) + count

        t1 = time.time()
        print(f"[TIMING] Pre-tokenization completed: {t1 - t0:.2f}s")
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


class Tokenizer:
    """
    A BPE (Byte Pair Encoding) tokenizer that encodes text into token IDs
    and decodes token IDs back into text.

    The tokenizer is initialized with a vocabulary and list of merges from
    BPE training, and optionally supports special tokens that are never split.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens.

        Args:
            vocab: The tokenizer vocabulary, a mapping from int (token ID)
                   to bytes (token bytes).
            merges: A list of BPE merges. Each item is a tuple of bytes
                    (token1, token2), representing that token1 was merged
                    with token2. Merges are ordered by order of creation.
            special_tokens: A list of string special tokens to add to the
                           vocabulary. These strings will never be split
                           into multiple tokens.
        """
        # 1. Keep the vocab and its reverse mapping (bytes->int)
        self.vocab = vocab
        self.vocab_reverse = {v: k for k, v in vocab.items()}

        # 2. Keep the merges as a mapping {(bytes, bytes): index}
        self.merges = {pair: idx for idx, pair in enumerate(merges)}

        # 3. Keep special tokens and append to vocab if not present
        self.special_tokens = special_tokens if special_tokens else []
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab_reverse:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_reverse[token_bytes] = new_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Class method that constructs and returns a Tokenizer from serialized
        vocabulary and merges files.

        Args:
            vocab_filepath: Path to the vocabulary JSON file. The file should
                           contain a mapping from token ID (as string) to
                           token bytes (as list of integers).
            merges_filepath: Path to the merges JSON file. The file should
                            contain a list of merge pairs, where each pair
                            is a list of two byte sequences (each as list of integers).
            special_tokens: A list of string special tokens to add to the
                           vocabulary.

        Returns:
            A Tokenizer instance constructed from the loaded vocabulary and merges.
        """
        # Load vocab from JSON: {token_id (str): hex_string}
        with open(vocab_filepath, "r") as f:
            vocab_data = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_data.items()}

        # Load merges from JSON: list of [hex1, hex2]
        with open(merges_filepath, "r") as f:
            merges_data = json.load(f)
        merges = [(bytes.fromhex(pair[0]), bytes.fromhex(pair[1])) for pair in merges_data]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text: The input text string to encode.

        Returns:
            A list of integer token IDs representing the encoded text.
        """
        # 1. Split the text on special tokens, then pre-tokenize
        # Build special token pattern (sort by length descending to match longer tokens first)
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            # Split but keep the delimiters (special tokens)
            parts = re.split(f"({special_pattern})", text)
        else:
            parts = [text]

        # Pre-tokenize each part (skip special tokens, they stay as-is)
        special_set = set(self.special_tokens)
        pre_tokenization_result: list[tuple[bytes, ...]] = []
        for part in parts:
            if not part:
                continue
            if part in special_set:
                # Special token: keep as single token
                pre_tokenization_result.append((part.encode("utf-8"),))
            else:
                # Regular text: apply GPT-2 pre-tokenization regex
                for match in PAT_COMPILED.finditer(part):
                    token_bytes = match.group().encode("utf-8")
                    # Convert to tuple of single bytes
                    pre_tokenization_result.append(tuple(bytes([b]) for b in token_bytes))

        # 2. For each item in the pre-tokenization, apply BPE merges
        merged_tokens: list[tuple[bytes, ...]] = []
        for item in pre_tokenization_result:
            new_item = item
            while True:
                # Find the pair with lowest merge index
                lowest_idx_in_merge = float("inf")
                pair_to_merge = None
                for idx in range(len(new_item) - 1):
                    pair = (new_item[idx], new_item[idx + 1])
                    if pair in self.merges and self.merges[pair] < lowest_idx_in_merge:
                        lowest_idx_in_merge = self.merges[pair]
                        pair_to_merge = pair

                # If no merge found, we're done with this item
                if pair_to_merge is None:
                    break

                # Merge all occurrences of pair_to_merge
                tmp_item: list[bytes] = []
                idx = 0
                while idx < len(new_item):
                    if idx < len(new_item) - 1 and (new_item[idx], new_item[idx + 1]) == pair_to_merge:
                        tmp_item.append(new_item[idx] + new_item[idx + 1])
                        idx += 2
                    else:
                        tmp_item.append(new_item[idx])
                        idx += 1
                new_item = tuple(tmp_item)

            merged_tokens.append(new_item)

        # 3. Translate all bytes to int based on the reverse mapping of vocab
        result: list[int] = []
        for token_tuple in merged_tokens:
            for token_bytes in token_tuple:
                result.append(self.vocab_reverse[token_bytes])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return
        a generator that lazily yields token IDs.

        This is required for memory-efficient tokenization of large files
        that cannot be directly loaded into memory.

        Args:
            iterable: An iterable of strings to encode (e.g., file handle
                     that yields lines).

        Yields:
            Integer token IDs one at a time.
        """
        # Use a generator: for each string, encode it and yield token IDs one by one
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: A list of integer token IDs to decode.

        Returns:
            The decoded text string. Invalid UTF-8 sequences are replaced
            with the Unicode replacement character.
        """
        # Translate token IDs to bytes and concatenate
        byte_pieces = [self.vocab[token_id] for token_id in ids]
        all_bytes = b"".join(byte_pieces)
        # Decode to string, replacing invalid UTF-8 with U+FFFD
        return all_bytes.decode("utf-8", errors="replace")
