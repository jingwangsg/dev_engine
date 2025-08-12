import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dev_engine.logging import logging as log

def draw_heatmap(tensor: torch.Tensor | np.ndarray, path: str = "example.png"):
    log.debug(f"Drawing heatmap to {path}")
    if isinstance(tensor, torch.Tensor):
        if torch.is_floating_point(tensor):
            tensor = tensor.float()
        tensor = tensor.detach().cpu().numpy()
    assert tensor.ndim == 2, "Heatmap must be 2D"
    plt.clf()

    sns.heatmap(tensor, cmap="viridis").get_figure().savefig(path)


def draw_histogram(tensor: torch.Tensor | np.ndarray, path: str = "example.png"):
    log.debug(f"Drawing histogram to {path}")
    if isinstance(tensor, torch.Tensor):
        if torch.is_floating_point(tensor):
            tensor = tensor.float()
        tensor = tensor.detach().cpu().numpy()
    plt.clf()

    sns.histplot(tensor).get_figure().savefig(path)


def draw_barplot(tensor: torch.Tensor | np.ndarray, path: str = "example.png"):
    log.debug(f"Drawing barplot to {path}")
    if isinstance(tensor, torch.Tensor):
        if torch.is_floating_point(tensor):
            tensor = tensor.float()
        tensor = tensor.detach().cpu().numpy()
    plt.clf()

    sns.barplot(x=np.arange(tensor.shape[0]), y=tensor).get_figure().savefig(path)
