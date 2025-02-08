from PIL.Image import Image
import torch
import numpy as np
from dev_engine import logging as log
from einops import rearrange
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt


def maybe_denormalize_tensor(tensor: torch.Tensor):
    if torch.is_floating_point(tensor):
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    return tensor


def maybe_denormalize_ndarray(array: np.ndarray):
    if np.issubclass(array.dtype.type, np.floating):
        if array.min() < 0:
            array = (array + 1) / 2
        array = (array * 255).clip(0, 255).astype(np.uint8)
    return array


def write_video(
    tensor: torch.Tensor | np.ndarray,
    input_format: str = "t h w c",
    path="example.mp4",
    fps=1,
):
    log.debug(f"Writing video to {path}")
    if input_format != "t h w c":
        tensor = rearrange(tensor, f"{input_format} -> t h w c")
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    tensor = maybe_denormalize_tensor(tensor)

    torchvision.io.write_video(path, tensor, fps=fps)


def write_image(
    tensor: Image | np.ndarray | torch.Tensor,
    path: str = "example.png",
    input_format="h w c",
):
    log.debug(f"Writing image to {path}")
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
        if input_format != "h w c":
            tensor = rearrange(tensor, f"{input_format} -> h w c")
        tensor = maybe_denormalize_ndarray(tensor)
        img = Image.fromarray(tensor)
    elif isinstance(tensor, np.ndarray):
        if input_format != "h w c":
            tensor = rearrange(tensor, f"{input_format} -> h w c")
        tensor = maybe_denormalize_ndarray(tensor)
        img = Image.fromarray(tensor)
    else:
        assert isinstance(tensor, Image)
        img = tensor

    img.save(path)


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
