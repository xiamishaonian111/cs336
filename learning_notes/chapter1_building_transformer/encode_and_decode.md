The encode process:
1. Split the text on special tokens (keep them), then pre-tokenize
2. For each item in the pre-tokenization, apply BPE merges
   - Find the pair with lowest merge index (to break tie when multiple bytes pairs can be merged in this round)
   - Do the merge
   - Repeat until we found no merge candidates in this round, or the item has been merged into a single `bytes` object
3. Translate all bytes into int

When encoding a large file, we need to use a buffered streaming mechanism. To do this, we should implement an 'encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]' in the Tokenizer, which can be implemented with a simple generator in Python.
