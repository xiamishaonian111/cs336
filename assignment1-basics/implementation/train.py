"""Training script for TransformerLM."""

import argparse
import os
import time

import numpy as np
import torch

from implementation.adamw import AdamW
from implementation.experiment_logger import ExperimentLogger
from implementation.checkpointing import load_checkpoint, save_checkpoint
from implementation.cross_entropy import cross_entropy
from implementation.get_batch import get_batch
from implementation.gradient_clipping import clip_gradient
from implementation.lr_schedule import get_lr_cosine_schedule
from implementation.transformer_lm import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TransformerLM.")

    # Model architecture
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # LR schedule
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--max_iters", type=int, required=True)
    parser.add_argument("--min_lr", type=float, default=0.0)

    # Training
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Paths
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to save metrics.jsonl and config.json")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def load_dataset(path: str) -> np.ndarray:
    """Load a tokenized dataset as a memory-mapped numpy array."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    """Compute average validation loss over num_batches."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            # Reshape for cross_entropy: (batch*seq, vocab) and (batch*seq,)
            loss = cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main() -> None:
    args = parse_args()

    # ---------- Dataset loading ----------
    train_data = load_dataset(args.train_path)
    val_data = load_dataset(args.val_path) if args.val_path else None

    # ---------- Model & optimizer ----------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ---------- Checkpoint resume ----------
    start_iter = 0
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            start_iter = load_checkpoint(ckpt_path, model, optimizer)
            print(f"Resumed from checkpoint at iteration {start_iter}")

    # ---------- Wandb init ----------
    if args.wandb_project:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    # ---------- Experiment logger ----------
    logger = None
    if args.log_dir:
        logger = ExperimentLogger(args.log_dir)
        logger.start(config=vars(args))

    # ---------- Training loop ----------
    model.train()
    for step in range(start_iter + 1, args.max_iters + 1):
        t0 = time.time()

        # LR schedule
        lr = get_lr_cosine_schedule(
            t=step,
            alpha_max=args.lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_iters,
            T_c=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)

        # Loss — reshape for cross_entropy((batch*seq, vocab), (batch*seq,))
        loss = cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0.0:
            clip_gradient(model.parameters(), max_norm=args.grad_clip)

        # Optimizer step
        optimizer.step()

        dt = time.time() - t0

        # ---------- Logging ----------
        if step % args.log_interval == 0:
            print(f"step {step} | loss {loss.item():.4f} | lr {lr:.6f} | {dt*1000:.0f}ms")
            if logger:
                logger.log(step=step, train_loss=loss.item(), lr=lr, step_time_ms=dt * 1000)
            if args.wandb_project:
                import wandb

                wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/step_time_ms": dt * 1000}, step=step)

        # ---------- Validation ----------
        if val_data is not None and step % args.val_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, args.device, args.val_batches)
            print(f"step {step} | val_loss {val_loss:.4f}")
            if logger:
                logger.log(step=step, val_loss=val_loss)
            if args.wandb_project:
                import wandb

                wandb.log({"val/loss": val_loss}, step=step)

        # ---------- Checkpoint saving ----------
        if args.checkpoint_dir and step > 0 and step % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"Saved checkpoint at iteration {step}")

    # Save final checkpoint
    if args.checkpoint_dir:
        ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
        save_checkpoint(model, optimizer, args.max_iters, ckpt_path)
        print("Saved final checkpoint")

    if logger:
        logger.finish()

    if args.wandb_project:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
