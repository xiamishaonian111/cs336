import math


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Args:
        t: Current iteration (step number)
        alpha_max: Maximum learning rate
        alpha_min: Minimum learning rate
        T_w: Number of warmup iterations
        T_c: Total number of cosine cycle iterations

    Returns:
        Learning rate at iteration t

    Schedule:
        - t < T_w:      Linear warmup from 0 to alpha_max
        - T_w <= t <= T_c: Cosine decay from alpha_max to alpha_min
        - t > T_c:      Constant at alpha_min
    """
    if t < T_w:
        return t / T_w * alpha_max
    elif t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min
