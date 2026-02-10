"""Generate text from a trained TransformerLM checkpoint."""

import argparse
import torch

from implementation.checkpointing import load_checkpoint
from implementation.tokenizer import Tokenizer
from implementation.transformer_lm import TransformerLM
from implementation.decode import decode


def main():
    parser = argparse.ArgumentParser(description="Generate text from a TransformerLM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, default="implementation/output/tinystories_vocab.json")
    parser.add_argument("--merges_path", type=str, default="implementation/output/tinystories_merges.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Model architecture (must match the checkpoint)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"])
    eos_token_id = tokenizer.vocab_reverse.get("<|endoftext|>".encode("utf-8"), None)

    # Build model and load checkpoint
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_ids)}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print("=" * 60)

    # Generate
    output_ids = decode(
        model=model,
        token_ids=prompt_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        device=args.device,
    )

    # Decode and print
    output_text = tokenizer.decode(output_ids)
    print(output_text)
    print("=" * 60)
    print(f"Total tokens generated: {len(output_ids) - len(prompt_ids)}")


if __name__ == "__main__":
    main()
