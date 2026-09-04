#!/Users/donyin/miniconda3/envs/imperial/bin/python

"""
get the total number of trainable parameters from the model
"""

from src.analysis.run_loader import RunLoader


def get_num_parameters(run_loader: RunLoader):
    total_params = sum(p.numel() for p in run_loader.model.parameters())
    trainable_params = sum(
        p.numel() for p in run_loader.model.parameters() if p.requires_grad
    )
    return total_params, trainable_params


if __name__ == "__main__":
    pass
