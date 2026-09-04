# Results
SIGMOID_SMOOTHED - not generalisable

# 🎭 Masking Mechanisms
The mask folder houses a collection of utilities and methods tailored for masking the raw logits produced from convolutional and pooling outputs. Predominantly utilized within the context of flexlayers, these masking functions serve as the bridge between the raw logits and the binary masks.

Note: Usage:
Each is a function that takes one logits and outputs a tesnsor of the same shape.

e.g.,
```python
logits = sigmoid_smoothed(logits)
```

## Sigmoid

## Sigmoid Scaled by 50
- Baseline method 2 (the previous method)

## Sigmoid Smoothed
- For "SIGMOID_SMOOTHED", a modified sigmoid function is used. The function is smoothed using temperature annealing to differentiate a discrete distribution.

## Straight Through Estimator (STE)
- For "STE", the logits are thresholded at 0.5 to make a binary mask. Gradients pass through unchanged during the backward pass.

## STE with Sigmoid
- For "STE_SIGMOID", the forward pass is similar to "STE". During the backward pass, gradients are modified by the gradient of a sigmoid function applied to the logits.