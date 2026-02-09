"""
This script contains a single class: VoxelGlobalAttention. The class is designed to use attention mechanisms on two input voxel representations to produce an output tensor.

Classes:
    - VoxelGlobalAttention:
        This class implements a transformer encoder that takes two tensors of shape (batch, channel, height, width) as inputs, converts each tensor to a vector representation using VoxelToVectorConv, and then concatenates the two tensors along with their positional embeddings. The combined tensor is passed through a transformer encoder, and the resulting encoded tensor is then transformed to compute the logits using a linear layer. It has one primary method: forward, which performs the described operations.

Note:
    The actual transformation of voxel to vector is abstracted out and is assumed to be defined in the VoxelToVectorConv class imported from src.modules.logits._utils.
"""

import math
import torch
import torch.nn as nn
from src.modules.logits._utils import VoxelToVectorConv


# --------------------------------------------------------------------------------
class VoxelGlobalAttention(nn.Module):
    def __init__(
        self,
        d_model=8,
        nhead=4,
        num_layers=2,
        out_dimensions=(8, 3, 32, 32),
        config=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dimensions = out_dimensions
        self.batch_size = out_dimensions[0]
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=config.get("dropout", 0.2),
            dim_feedforward=2048,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        # -------- setup the voxel to vector layers and positional embedding --------
        self.voxel_to_vector_1 = VoxelToVectorConv(additional_dim=d_model)
        self.voxel_to_vector_2 = VoxelToVectorConv(additional_dim=d_model)
        self.pos_embedding = nn.Embedding(
            math.prod(out_dimensions[1:]), d_model
        )  # excluding batch dimension
        self.linear_layer = nn.Linear(2 * d_model, 1)  # excluding batch dimension

    def add_positional_embeddings(self, x):
        positions = (
            torch.arange(0, x.size(1), device=x.device)
            .unsqueeze(0)
            .repeat(x.size(0), 1)
        )
        return x + self.pos_embedding(positions)

    def forward(self, in_tensor_1, in_tensor_2):
        # -------- convert the tensors to vectors
        in_tensor_1 = self.voxel_to_vector_1(in_tensor_1)
        in_tensor_2 = self.voxel_to_vector_2(in_tensor_2)

        error_1 = f"vectorized tensor shape should be (batch, channel, height, width, d_model). Got {in_tensor_1.shape}"
        error_2 = f"tensors must be of the same shape"
        assert len(in_tensor_1.shape) == 5, error_1
        assert in_tensor_1.shape == in_tensor_2.shape, error_2

        in_tensor_1 = in_tensor_1.reshape(self.batch_size, -1, self.d_model)
        in_tensor_1 = self.add_positional_embeddings(in_tensor_1)

        in_tensor_2 = in_tensor_2.reshape(self.batch_size, -1, self.d_model)
        in_tensor_2 = self.add_positional_embeddings(in_tensor_2)

        tensor_concate = torch.cat([in_tensor_1, in_tensor_2], dim=1)

        # -------- reshape and feed to transformers -------
        transformer_output = self.transformer_encoder(
            tensor_concate
        )  # it is expecting (batch, seq, feature); checked
        transformer_output = transformer_output.reshape(
            *self.out_dimensions, self.d_model * 2
        )
        logits = self.linear_layer(transformer_output)
        logits = logits.reshape(*self.out_dimensions)
        return logits


if __name__ == "__main__":
    # Example usage
    dummy_conv = torch.randn(8, 3, 32, 32)
    dummy_pool = torch.randn(8, 3, 32, 32)

    voxel_transformer = VoxelGlobalAttention(
        d_model=16, nhead=2, num_layers=1, out_dimensions=(8, 3, 32, 32)
    )
    voxel_transformer(dummy_conv, dummy_pool)
    # print(logits.shape)
