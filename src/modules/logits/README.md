# Results
"SPATIAL_ATTENTION",  # plain spatial attention / no advantage over weighted s / see below
"SPATIAL_ATTENTION_CMP",  # no advantage over weighted sum in valid accuracy / attack auc
"UNET",  # it has all the benefits, but simply too computationally expensive
"SPATIAL_ATTENTION_GATED",  # performance is not good
"SPATIAL_ATTENTION_PRE_MUL",  # performance is not good
"SPATIAL_ATTENTION_PRE_SUM",  # performance is not good / dumb version of weighted sum

1 / 3 spatial attention blocks does not offer much difference in anything

# 📊 Logits Methods
The `logits` folder provides methods that are used in generating tensors of logits from the convolutional and pooling outputs. This is only used for flexlayers. The logits are the raw data that needs to be masked by some form of activation functions or regulations to for the binary masks used to determine the final output.

## Thresholding
- The baseline method. Applying a multuplier of 50 to make the sigmoid very sharp. This is now suspected to cause issues in the training dynamics (gradient masking) but yet tested.

## Sparial Attention
- Involves using multiple spatial_attention_blocks on the convolution tensor to get logits.

## Spatial Attention Sum
[spatial_attentiom_sum.py]
- Perform element-wise addition of the two tensors, and then apply a series of convolutional layers

## Spatial Attention Multiplication
[spatial_attentiom_mul.py]
- Perform element-wise multiplication of the two tensors, and then apply a series of convolutional layers

## Spatial Attention Channel-wise MaxPool
- Process the conv and pool separately using conv blocks and do a channel-wise softmax pooling between them

## Spatial Attention Gated
- Process the conv and pool separately using conv blocks and do a channel-wise sigmoid gate to sum them up

## Spatial Attention Weighted Sum
- Process the conv and pool separately using conv blocks and add a trainable parameter for a weighted sum

## Unet Block
- Treat the channel dimension of the pool and conv and the depth dimension and so the two tensors are made 3D. The two 3D tensors are then fed into a 3D unet, with the pool/conv being treated as the channel dimension. The 3D UNet then outputs a tensor of size (b, c=1, d, h, w).

## Transformer Positional
- **Voxel-to-Vector Conversion**: Convert each voxel representation into a vector using 3D convolution.
- **Transformer Encoding**: Feed vector representations into a transformer encoder with a classification token. Process the encoder's output via a linear layer to produce logits.

## Transformer Global
- Two input tensors (different voxel modalities) are initially converted into vector forms using 3D convolutions.
- Vectorized representations are concatenated, and positional information is embedded. 
- The concatenated tensor is processed through a transformer encoder, capturing long-range dependencies.
- The encoder's output is processed to compute logits for further tasks.
