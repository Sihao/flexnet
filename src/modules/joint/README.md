# 🔗 Joint Methods
This folder contains functions that take the conv and pool tensors and produce an output tensor with the same shape as either of the two input tensors. This bypasses the standard process of the "logits - masking" two steps.

## Channel-wise Max-Pooling
Simply use "max" to choose between "max" and "pool". The gradient flows through any channel that is larger than the other, and only that specific channel is optimized. Note that this is not the same as backpropagating on "argmax"; "argmax" is not differentiable.