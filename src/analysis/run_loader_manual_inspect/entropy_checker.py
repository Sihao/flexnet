"""
input a folder,
confirm that this folder contains a .pth in recrusive search
make a model according to the config
load the modal
test the model on the 1000 test image
plot entropy
"""

from tqdm import tqdm
from pathlib import Path
import plotly.graph_objects as go
from natsort import natsorted
import json
from src.utils.hyperparams.settings import AttrDict
import torch
import random
from src.flex_neurons.utils.device import select_device
from torch.utils.data import DataLoader
from plotly.subplots import make_subplots
import numpy as np
from dash import Dash, html, dash_table
from dash_slicer import VolumeSlicer
from src.analysis.run_loader import RunLoader
from src.flex_neurons.data.dataset_select import get_dataset_obj
from src.flex_neurons.data.dataset_subset import create_random_subset


class EntropyChecker:
    def __init__(self, run_loader: RunLoader):
        self.run_loader = run_loader
        self.run_folder = run_loader.run_folder
        self.device = select_device()
        print(f"Using device: {self.device}")
        # self._search_files()
        self._main()

    # def _search_files(self):  # later just use that in the RunLoader
    #     self.config = list(self.run_folder.rglob("configurations.json"))[0]
    #     self.config = json.load(self.config.open("r"))
    #     self.config = AttrDict(self.config)
    #     assert self.config.get("use_flex"), "This model does not use Flex2D layers."

    def _main(self):
        dataset = get_dataset_obj(self.config["dataset"], "TEST")
        dataset = create_random_subset(dataset, 128)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        # initialize accumulators for each layer
        layer_activations_dict = {}

        with torch.no_grad():
            for image, _ in tqdm(dataloader):
                image = image.to(self.device)
                self.run_loader.model(image)
                layer_cp_matrices = self.run_loader.model._get_cp_id_matrices()

                for idx, item in enumerate(layer_cp_matrices):
                    if f"layer_{idx}" not in layer_activations_dict:
                        layer_activations_dict[f"layer_{idx}"] = []
                    layer_activations_dict[f"layer_{idx}"].append(
                        item.squeeze(0).float()
                    )

        # -------- calculate the entropy for each layer --------
        # this results in some 3D tensors; which needs to be plotted, maybe in plotly dash
        # shannon entropy = -p * log2(p) - (1-p) * log2(1-p)
        # i am using a normalized shannon entropy
        entropy_dict = {}
        for layer_name, activation_list in layer_activations_dict.items():
            stacked_activations = torch.stack(activation_list, dim=0)
            activation_prob = stacked_activations.mean(dim=0)
            activation_prob = torch.clamp(activation_prob, 1e-8, 1 - 1e-8)
            entropy = -activation_prob * torch.log2(activation_prob) - (
                1 - activation_prob
            ) * torch.log2(1 - activation_prob)
            entropy = torch.nan_to_num(entropy, nan=0.0)
            entropy_dict.update({layer_name: entropy})

        self.entropies = entropy_dict


class DashEntropyVisualiser:
    def __init__(self, checker):
        assert isinstance(checker, EntropyChecker), "Expecting EntropyChecker object."
        self.checker = checker
        self.app = Dash(__name__)

    def visualize_layer(self, layer_name):
        if layer_name not in self.checker.entropies:
            raise ValueError(f"Layer {layer_name} not found in EntropyChecker.")
        volume = self.checker.entropies[layer_name].numpy()

        slicer0 = VolumeSlicer(self.app, volume, axis=0)
        slicer1 = VolumeSlicer(self.app, volume, axis=1)
        slicer2 = VolumeSlicer(self.app, volume, axis=2)

        slicer0.graph.config["scrollZoom"] = False
        slicer1.graph.config["scrollZoom"] = False
        slicer2.graph.config["scrollZoom"] = False

        # html.H1(f"Flex Num: {layer_name}"),
        div_main = html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "33% 33% 33%",
            },
            children=[
                html.Div([slicer0.graph, html.Br(), slicer0.slider, *slicer0.stores]),
                html.Div([slicer1.graph, html.Br(), slicer1.slider, *slicer1.stores]),
                html.Div([slicer2.graph, html.Br(), slicer2.slider, *slicer2.stores]),
            ],
        )
        div_main = html.Div(
            style={
                "display": "flex",
                "flexDirection": "column",
            },
            children=[html.H1(f"Flex Num: {layer_name.split('_')[-1]}"), div_main],
            id=layer_name,
        )

        return div_main

    def visualize_all_layers(self):
        divs = [
            self.visualize_layer(layer_name)
            for layer_name in self.checker.entropies.keys()
        ]
        container = html.Div(
            style={"display": "flex", "flexDirection": "column"}, children=[*divs]
        )
        self.app.layout = container
        self.app.run(debug=True, dev_tools_props_check=False)


if __name__ == "__main__":
    loader = RunLoader(Path("/Users/donyin/Desktop/runs/87b45fd6"))
    checker = EntropyChecker(loader)  # 87b45fd6
    DashEntropyVisualiser(checker).visualize_all_layers()
