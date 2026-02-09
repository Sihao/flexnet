#!/Users/donyin/miniconda3/envs/imperial/bin/python -m pdb
import numpy as np
from pathlib import Path
import warnings, pickle, json, torch
import matplotlib.pyplot as plt
from src._pyhessian.pyhessian import hessian  # Hessian computation
from src._pyhessian.density_plot import get_esd_plot
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_balanced_subset, create_random_subset
from copy import deepcopy

from torch.utils.data import DataLoader
from src.analysis.run_loader import RunLoader
from src.utils.device import select_device

warnings.filterwarnings("ignore")


# replace the dataset with balanced dataset
# -------- helpers --------
def perturbe_model(model_original, model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(
        model_original.parameters(), model_perb.parameters(), direction
    ):
        m_perb.data = m_orig.data + alpha * d
    return model_perb


# -------- main functions --------
class HessianAnalysis:
    def __init__(self, run_loader: RunLoader, batch_size=16):
        self.run_loader, self.batch_size = run_loader, batch_size

        self.prepare_databatch()
        self.get_hessian()

    # -------- preparation --------
    def prepare_databatch(self):
        dataset = self.run_loader.config.dataset
        dataset = get_dataset_obj(dataset, "TRAIN")
        dataset = create_random_subset(dataset, self.batch_size)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.input, self.target = next(
            iter(dataloader)
        )  # this should take the entire dataset
        self.input, self.target = self.input.to(self.run_loader.device), self.target.to(
            self.run_loader.device
        )

    def get_hessian(self):
        """the main function of getting the hessian computation object of the model"""
        hessian_computation = hessian(
            self.run_loader.model,
            self.run_loader.criterion,
            data_batch=(self.input, self.target),
        )
        self.hessian_computation = hessian_computation

    # -------- main functions --------
    def get_top_n_eigenvalues(self, save_dir: Path, top_n=8):
        save_dir.mkdir(parents=True, exist_ok=True)
        top_eigenvalues, self.top_eigenvector = self.hessian_computation.eigenvalues(
            top_n=top_n
        )

        np.save(save_dir / f"hessian_top_eigen_values.npy", top_eigenvalues)

        with open(
            save_dir / f"hessian_top_eigen_vectorss.pkl", "wb"
        ) as f:  # list[list[torch.Tensor]]
            pickle.dump(self.top_eigenvector, f)

    def get_trace_of_hessian(self, save_dir: Path, num_iter=1):
        """[note]
        this makes the data for the historgram plot of trace

        this function is semi-random, thus needs num_iter to capture some variance; when load:
            density_eigen = np.load("density_eigen.npy")
            density_weight = np.load("density_weight.npy")

        [plot]
            get_esd_plot(density_eigen, density_weight, save_as=save_as)
        """
        # clear the device
        torch.cuda.empty_cache()
        save_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_iter):
            save_dir_iter = save_dir / f"hessian_iter_{i}"
            save_dir_iter.mkdir(parents=True, exist_ok=True)
            trace = self.hessian_computation.trace()
            density_eigen, density_weight = self.hessian_computation.density()
            np.save(save_dir_iter / f"density_eigen.npy", density_eigen)
            np.save(save_dir_iter / f"density_weight.npy", density_weight)
            np.save(save_dir_iter / f"trace_mean.npy", np.mean(trace))

    def dump_configurations(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_loader.run_folder / "configurations.json", "r") as f:
            configurations = json.load(f)
        with open(save_dir / "configurations.json", "w") as f:
            json.dump(configurations, f, indent=4)


if __name__ == "__main__":
    from src.utils.function_timer import CodeProfiler

    profiler = CodeProfiler()
    profiler.start()
    # -------- [ start of the function of interests ] --------
    for run in [
        # "e0f9adcf",  # original vgg16
        # "5331087d",  # cmp
        # "fa3a501b",  # original flex nbn
        "05560acf",  # hard sigmoid
    ]:
        run_loader = RunLoader(
            f"/Users/donyin/Desktop/experiment/{run}", whether_load_checkpoint=False
        )
        hessian_analysis = HessianAnalysis(run_loader)
        hessian_analysis.get_top_n_eigenvalues(save_dir=Path("_hessian", run), top_n=3)
        hessian_analysis.get_trace_of_hessian(
            save_dir=Path("_hessian", run), num_iter=1
        )
        hessian_analysis.get_loss_landscape_along_eigenvectors(
            save_as=Path(f"_hessian/{run}/hessian_loss_landscape.png"), num_samples=21
        )

        print(f"{run} - successful")
    # -------- [ end of the function of interests ] --------
    profiler.end()
