import torch
from torch.utils.data import Dataset
import numpy as np
import data_io


class DeepInterpolationDataset(Dataset):
    def __init__(self, tiff_path, frame_window_N):
        """
        Args:
             tiff_path (str): Path to tiff stack.
             frame_window_N (int): Number of frames before/after to use as input.
        """
        self.tiff_path = tiff_path
        self.N = frame_window_N

        print(f"Loading tiff stack from {tiff_path}...")
        # Shape: (T, H, W)
        self.frames = data_io.load_tiff_stack(tiff_path)
        self.T, self.H, self.W = self.frames.shape
        print(f"Loaded stack with shape {self.frames.shape}")

        # Valid samples are those where we can take N frames before and after
        # Indices valid for CENTER frame: [N, T - N - 1]
        # Length = (T - N - 1) - N + 1 = T - 2N
        self.length = self.T - 2 * self.N

        if self.length <= 0:
            raise ValueError(f"Tiff stack (T={self.T}) too short for window N={self.N}")

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        # i maps to index in valid range
        # Center frame index
        idx = i + self.N

        # Pre-frames: [idx-N, ..., idx-1]
        pre = self.frames[idx - self.N : idx]

        # Post-frames: [idx+1, ..., idx+N]
        post = self.frames[idx + 1 : idx + self.N + 1]

        # Center frame target
        target = self.frames[idx]

        # Stack inputs along channel dimension
        # (N, H, W) + (N, H, W) -> (2N, H, W)
        input_stack = np.concatenate([pre, post], axis=0)

        # Convert to Tensor (Float32)
        input_tensor = torch.from_numpy(input_stack).float()
        target_tensor = torch.from_numpy(target).float().unsqueeze(0)  # (1, H, W)

        return input_tensor, target_tensor
