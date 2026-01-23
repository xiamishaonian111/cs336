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
    # more details here:
    # 1. the process for each step is: iterate over all pairs from pre-tokenization results -> find the most frequent pair -> add it to vocab, merges and update pre-tokenization result.
    # 2. we do not consider the scenario where there are no more pairs to merge before we reach the expected `vocab_size`
    # 3. when two pairs have the same frequency, break tie tie by preferring the lexi greater pair.

    # Initializa the list to store merges we've encountered.    
    merges: list[tuple[bytes, bytes]] = []
    # Repeat the merging process.
    for i in range(0, vocab_size - 256 - len(special_tokens)):
        pair_freq: dict[tuple[bytes, bytes], int] = {}
        # TODO 6.1: record frequency of every pair
        for token, freq in pre_tokenization.items():
            for j in range(len(token) - 1):
                pair = (token[j], token[j + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq

        # TODO 6.2: find the most frequent pair
        max_freq_pair: tuple[bytes, bytes] | None = None
        max_freq = 0
        for pair, freq in pair_freq.items():
            if freq > max_freq or (freq == max_freq and pair > max_freq_pair):
                max_freq = freq
                max_freq_pair = pair

        # TODO 6.3: add the merge to merges
        merges.append(max_freq_pair)

        # TODO 6.4: add the merged pair to vocab
        vocab[len(vocab)] = max_freq_pair[0] + max_freq_pair[1]

        # TODO 6.5: update pre-tokenization result by replacing merged pairs
        new_pre_tokenization: dict[tuple[bytes, ...], int] = {}
        merged_token = max_freq_pair[0] + max_freq_pair[1]
        for token, freq in pre_tokenization.items():
            # Skip merge logic if pair can't exist in this token
            if max_freq_pair[0] not in token or max_freq_pair[1] not in token:
                new_pre_tokenization[token] = new_pre_tokenization.get(token, 0) + freq
                continue
            # Replace all occurrences of the merged pair
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
            new_pre_tokenization[new_key] = new_pre_tokenization.get(new_key, 0) + freq
        pre_tokenization = new_pre_tokenization


    # TODO: Return the final vocabulary and merges list
    return vocab, merges
