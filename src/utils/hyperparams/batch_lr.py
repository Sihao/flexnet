#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
- batch size and learning rate has to be adjusted when the other changes
"""


def adjust_batch_learning_rate(
    objective: str,
    other_value: float,
    base_batch_learning_rate_ratio: tuple,
    method: str,
):
    """
    ---------------------------------------------
    :param objective: str, "batch" or "learning_rate"
    :param other_value: float, the value of the other value opposed to the objective
    :param base_batch_learning_rate_ratio: tuple, (base_batch_size, base_learning_rate)
    :param method: str, "LINEAR" or "ALEX"
    ---------------------------------------------
    Theory suggests that when multiplying the batch size by k, one should multiply the learning rate by sqrt(k) to keep the variance in the gradient expectation constant. See page 5 at A. Krizhevsky. One weird trick for parallelizing convolutional neural networks: https://arxiv.org/abs/1404.5997

    However, recent experiments with large mini-batches suggest for a simpler linear scaling rule, i.e multiply your learning rate by k when using mini-batch size of kN. See P.Goyal et al.: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour https://arxiv.org/abs/1706.02677
    """
    base_batch_size, base_learning_rate = base_batch_learning_rate_ratio

    if method == "LINEAR":
        if objective == "batch":
            k = other_value / base_learning_rate
            adjusted_value = base_batch_size * k
        elif objective == "learning_rate":
            k = other_value / base_batch_size
            adjusted_value = base_learning_rate * k

    elif method == "ALEX":
        if objective == "batch":
            k = (other_value / base_learning_rate) ** 2
            adjusted_value = base_batch_size * k
        elif objective == "learning_rate":
            k = other_value / base_batch_size
            adjusted_value = base_learning_rate * (k**0.5)

    else:
        raise ValueError(f"Adjustment method |{method}| not found")

    return adjusted_value


if __name__ == "__main__":
    batch_learning_rate_ratio = (320, 1e-3)
    adjusted_batch_size = adjust_batch_learning_rate(
        "batch", 1e-4, batch_learning_rate_ratio, "LINEAR"
    )
    adjusted_learning_rate = adjust_batch_learning_rate(
        "learning_rate", 12, batch_learning_rate_ratio, "LINEAR"
    )
    print(f"Adjusted batch size: {adjusted_batch_size}")
    print(f"Adjusted learning rate: {adjusted_learning_rate}")
