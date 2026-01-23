High level process:
1. Split on special tokens, get different chunks.
2. In each chunk, do pre-tokenization with regex (in GPT-2, r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""").
   - Why? directly merging bytes across the corpus may result in tokens that differ only in punctuation (e.g., dog! vs. dog.). These tokens would get completely different token IDs, even though they are likely to have high semantic similarity (since they differ only in punctuation).
   - To speed it up, we can do pre-tokenization in parallel.
3. Train BPE in several iterations, for each it does: (1) count the highest frequency pair from pre-tokenization result (no cross boundary), and (2) do the merging: update the vocabulary, the pre-tokenization result and pair frequency map (and any other intermediate data structures for further optimization)
   - Optimization 1: Incrementally update the pair frequency map. Note that we need to keep a reverse map from pair to token, so we know which tokens to update.
   - Optimization 2 (note implement since the current version is already good enough): besides the pair frequency map, use a heap to dynamically manage the max frequency of all pairs. Note lazy update should be used in production because heap does not support O(logN) update for a specific element.
