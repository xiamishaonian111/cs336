"""Decoding / text generation from a TransformerLM."""

import torch

from implementation.softmax import softmax
from implementation.transformer_lm import TransformerLM


def decode(
    model: TransformerLM,
    token_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """Generate a completion from a language model given a prompt.

    Args:
        model: A TransformerLM instance (already on the correct device).
        token_ids: List of integer token IDs representing the prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Softmax temperature for scaling logits before sampling.
            Lower = more greedy, higher = more random. Must be > 0.
        top_p: Nucleus sampling threshold in (0, 1]. Only the smallest set
            of tokens whose cumulative probability >= top_p are kept.
            1.0 disables top-p (uses full distribution).
        eos_token_id: If provided, stop generation when this token is sampled.
        device: PyTorch device string.

    Returns:
        The full sequence (prompt + generated tokens) as a list of ints.
    """
    model.eval()
    context = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context_length if needed
            input_ids = context[:, -model.context_length :]

            # Forward pass — logits shape: (1, seq_len, vocab_size)
            logits = model(input_ids)

            # Take logits for the last position: (vocab_size,)
            next_logits = logits[0, -1, :]
            
            # Divide logits by temperature before converting to probabilities.
            next_logits = next_logits / temperature
            p = softmax(next_logits, dim=-1)

            # Apply top-p (nucleus) sampling.
            sorted_p, sorted_indices = torch.sort(p, descending=True)
            cumulative_p = torch.cumsum(sorted_p, dim=-1)
            # Zero out tokens beyond the top-p threshold
            sorted_p[cumulative_p - sorted_p > top_p] = 0.0
            # Re-normalize
            sorted_p = sorted_p / sorted_p.sum()

            # Sample from the filtered sorted distribution
            sampled_index = torch.multinomial(sorted_p, num_samples=1)
            # Map back to original vocab index
            next_token = sorted_indices[sampled_index]

            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return context[0].tolist()
