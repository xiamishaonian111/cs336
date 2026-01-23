High level process:
1. Split on special tokens, get different chunks (prevents merges across document boundaries).
2. In each chunk, do pre-tokenization with regex (in GPT-2, r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""").
   - Why? Directly merging bytes across the corpus may result in tokens that differ only in punctuation (e.g., "dog!" vs "dog."). These tokens would get completely different token IDs, even though they have high semantic similarity.
   - To speed it up, we can do pre-tokenization in parallel.
3. Train BPE iteratively. Each iteration: (1) find the highest frequency pair from pre-tokenization result (no cross-boundary merges, use lexicographic tie-breaking for determinism), and (2) merge: update the vocabulary, pre-tokenization result, and pair frequency map.
   - Optimization 1: Incrementally update the pair frequency map. Keep a reverse map from pair to tokens (`pair_to_tokens`), so we only visit tokens that contain the merged pair, instead of iterating through all tokens.
   - Optimization 2 (not implemented since current version is good enough): Use a heap to find max frequency pair in O(log P) instead of O(P). Requires lazy deletion since heaps don't support O(log N) updates for arbitrary elements.
