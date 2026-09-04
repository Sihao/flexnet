# Note
- Not used anymore as this just too far away from the point of the interest.
- also it didn't work very well in terms of model performance 

# Loss Functions for Logits Control
This folder contains a set of customized loss functions specifically designed to control the behavior and dynamics of model logits during training. Usually loss functions primarily focus on reducing the difference between predicted and ground-truth labels. However, in many modern deep learning applications, we may want more control over the logits' behavior beyond just fitting to the ground truth. This is where our customized loss functions come into play.

- **Binariness Control**: Encourages the logits to take extreme values (closer to 0 or 1) rather than values around 0.5.
- **Local Dynamics Control**: Apply penalties or rewards based on the local properties of logits, promoting specific desired characteristics.

## Exponential with Coefficient
[loss vs coefficient](peak_at_half.png)

## Clamped Binary
[binary loss](clamped_binary.png)