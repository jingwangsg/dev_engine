from PIL.Image import Image
import torch
import numpy as np
from dev_engine import logging as log
from einops import rearrange
import torchvision


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
    if input_format != "t h w c":
        tensor = rearrange(tensor, f"{input_format} -> t h w c")
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    tensor = maybe_denormalize_tensor(tensor)

    torchvision.io.write_video(path, tensor, fps=fps)


def write_image(
    tensor: Image | np.ndarray | torch.Tensor, path: str, input_format="h w c"
):
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
