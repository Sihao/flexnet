"""
This script contains two classes: VoxelToVector and VoxelTransformer, used together for making logits from two voxel representations. The logits are then used to compute the binary mask that maps the conv / pool decision.

Classes:

    - VoxelTransformer:
        This class implements a transformer encoder that takes two tensors of shape (channel, height, width, d_model) as inputs, concatenates them along with a classification token, and passes them through a transformer encoder. The encoded representation is then used to compute logits through a linear layer. It has one method: forward, which performs the described operations.
"""

import torch
import torch.nn as nn
from src.modules.logits._utils import VoxelToVectorConv


# --------------------------------------------------------------------------------
class VoxelTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model=768,
        nhead=8,
        num_layers=5,
        out_dimensions=(8, 3, 32, 32),
        config=None,
    ):
        super().__init__()
        self.out_dimensions = out_dimensions
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=config.get("dropout", 0.2),
            dim_feedforward=2048,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.classification_token = nn.Parameter(
            torch.randn(*out_dimensions, 1, d_model)
        )
        self.linear_layer = nn.Linear(d_model, 1)

        # -------- setup the voxel to vector layers --------
        self.voxel_to_vector_1 = VoxelToVectorConv(additional_dim=d_model)
        self.voxel_to_vector_2 = VoxelToVectorConv(additional_dim=d_model)

    def forward(self, in_tensor_1, in_tensor_2):
        # -------- convert the tensors to vectors
        in_tensor_1 = self.voxel_to_vector_1(in_tensor_1)
        in_tensor_2 = self.voxel_to_vector_2(in_tensor_2)

        error_1 = f"vectorized tensor shape should be (batch, channel, height, width, d_model). Got {in_tensor_1.shape}"
        error_2 = f"tensors must be of the same shape"
        assert len(in_tensor_1.shape) == 5, error_1
        assert in_tensor_1.shape == in_tensor_2.shape, error_2

        # -------- concatenate the tensors
        cls_with_tensors = torch.cat(
            [
                self.classification_token,
                in_tensor_1.unsqueeze(-2),
                in_tensor_2.unsqueeze(-2),
            ],
            dim=4,
        )

        # -------- reshape the combined tensor to feed into the transformer encoder
        cls_with_tensors = cls_with_tensors.reshape(
            -1, 3, self.d_model
        )  # it is expecting (batch, seq, feature); checked

        # -------- get results from the transformer --------
        transformer_output = self.transformer_encoder(cls_with_tensors)
        classification_output = transformer_output[
            :, 0, :
        ]  # since its the first on the token channel
        logits = self.linear_layer(classification_output)
        logits = logits.reshape(*self.out_dimensions)
        return logits


if __name__ == "__main__":
    # -------- example usage
    dummy_conv = torch.randn(8, 3, 32, 32)
    dummy_pool = torch.randn(8, 3, 32, 32)

    voxel_transformer = VoxelTransformerEncoder(
        d_model=768, nhead=8, num_layers=5, out_dimensions=(8, 3, 32, 32)
    )
    logits = voxel_transformer(dummy_conv, dummy_pool)

    print(logits.shape)
