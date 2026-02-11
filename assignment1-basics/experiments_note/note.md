1. Hyperparameter sweep over (we used low-resource version)

1.1. learning rate (with batch size as 32)
- we've explored those learning rates: "1e-4" "3e-4" "1e-3" "3e-3" "1e-2" "3e-2" "1e-1" "3e-1" "1".
- the learning rate with lowest val loss is 3e-3 (1.61), and 1e-3 is pretty close (1.65).
- the edge of stability is 3e-2 => 3e-2 is still stable, while 1e-1 is not.

1.2. batch size
- we've explored those batch sizes: 1, 2, 4, 8, 32, 64. We've OOM'ed at 64.
- training time decrease explicitly from 1->2->4. After BS=4, the training time decrease is not explicit anymore.
- smaller batch sizes have less final val loss, but the difference is small (especially if we tune the LR for each different batch size) and may not be a factor we have to consider in our tradeoff. 
- NOTE: searching online, if we have BS larger than **critical batch size**, that might greatly increase the val loss:
-- Below critical batch size: each gradient step is noise-dominated. Doubling batch size roughly halves the steps needed. Batch size choice barely matters for the final loss — you're trading steps for parallelism at ~equal compute efficiency.
-- Above critical batch size: Gradients are already well-estimated. Doubling batch size does NOT halve the steps needed.
- **Process should follow in practice:** (1) Start from hardware constraints (2) find where throughput saturates (3) compare it with critical batch size (below => use it; above => make a tradeoff between training wall time and final val loss)

2. Ablations

2.1. Remove RMSNorms
- most training have been divergent
- the training with very small LR is still ok (still worse than previously). But once LR becomes a little bit higher, the training would divergent.
- RMSNorm is **super critical** for training stability, for both forward and backward passes.

2.2. pre-norm => post-norm
- Converge slower, optimal LR is smaller. Minimal val loss (from different LRs) is larger.
- That's probably because the residual connection is negatively impacted. Pre-norm creates a **clean residual gradient highway**.
- It's harder to train stably with post-norm (would need learning rate warmup, initialization, etc.)

3. position embeddingshe
3.1. NoPE
- Has less impact in earlier runs, larger impact in later runs. Get larger final val loss.

4. SwiGLU => SiLU
- Both still converges.
- Training with SwiGLU converges faster, and get better final val loss.

5. Train on OpenWebText
- Its converge is slower than TinyStories, and final val loss is larger. Explanation: (1) the model is not large enough (2) tokens to processed is not large enough. Those two are large enough to get a small KL divergence (i.e. to learn the distribution) in TinyStories, but not OpenWebText.
- The fluency is worse than that model trained from TinyStories. The reason is as explained above: we do not have a large enough model and enough steps to train from a larger and more complicated data set (OpenWebText). The model learned from OWT just guesses some high frequent words (probably to reduce cross entropy, in the limited number of steps) but has no clear structure in the output.
