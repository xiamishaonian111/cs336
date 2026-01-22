"""BPE (Byte Pair Encoding) tokenizer training implementation."""

from __future__ import annotations

import os


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
    # Read the input file as bytes
    with open(input_path, "rb") as f:
        data = f.read()

    # TODO: Split the input file content to different chunks based on special tokens, by using re.split with "|".join(special_tokens).

    # TODO: In each chunk, separately do a regex-based pre-tokenization with r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # TODO: Merge the pre-tokenization result from different chunks

    # TODO: Initialize vocabulary with base byte tokens (0-255)
    vocab: dict[int, bytes] = {}

    # TODO: Add special tokens to vocabulary

    # TODO: Perform BPE training from pre-tokenization result - iteratively merge most frequent pairs
    merges: list[tuple[bytes, bytes]] = []
    for i in range(0, vocab_size - 256 - len(special_tokens)):
        pass

    # TODO: Return the final vocabulary and merges list
    return vocab, merges
