"""BPE (Byte Pair Encoding) tokenizer training implementation."""

from __future__ import annotations

import os
import re
import regex


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
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
    # Read the input file as string
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    # TODO 2: Do a regex-based pre-tokenization with r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".
    # more details here:
    # 1. pre_tokenization result has to be in the format of dict[bytes, int]. We do not store the strings, but store separate bytes.
    # 2. we should use re.finditer to avoid storing the pre-tokenized words as we construct your mapping from pre-tokens to their counts.
    # 3. split on special tokens before pre-tokenization: use re.split with "|".join(special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Split on special tokens to prevent merges across document boundaries
    special_token_pattern = "|".join(re.escape(token) for token in special_tokens)
    chunks = re.split(special_token_pattern, data)

    # Pre-tokenize each chunk and count frequencies
    pre_tokenization: dict[tuple[bytes, ...], int] = {}
    for chunk in chunks:
        for match in regex.finditer(PAT, chunk):
            token_bytes = match.group().encode("utf-8")
            # Convert to tuple of single-byte tokens for BPE processing
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            pre_tokenization[token_tuple] = pre_tokenization.get(token_tuple, 0) + 1

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
    pair_freq: dict[tuple[bytes, bytes], int] = {}
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for token, freq in pre_tokenization.items():
        for j in range(len(token) - 1):
            pair = (token[j], token[j + 1])
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
            if pair not in pair_to_tokens:
                pair_to_tokens[pair] = set()
            pair_to_tokens[pair].add(token)

    # Initialize the list to store merges
    merges: list[tuple[bytes, bytes]] = []

    # Repeat the merging process
    for _ in range(vocab_size - 256 - len(special_tokens)):
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
