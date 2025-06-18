import torch
import numpy as np
from PIL import ImageDraw

rainbow_palette = [
    (255, 0, 0),    # Red
    (255, 127, 0),  # Orange
    (255, 255, 0),  # Yellow
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (75, 0, 130),   # Indigo
    (148, 0, 211),  # Violet
    (255, 105, 180), # Hot Pink
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192), # Silver
    (128, 128, 128), # Gray
    (255, 215, 0),  # Gold
    (210, 105, 30)  # Chocolate
]

rainbow_color_idx = 0

def get_rainbow_color():
    global rainbow_color_idx
    color = rainbow_palette[rainbow_color_idx]
    rainbow_color_idx = (rainbow_color_idx + 1) % len(rainbow_palette)
    return color

def draw_lines(rgb, xy, linewidth=2, color=get_rainbow_color):
    """
    Draw lines on an RGB image using specified coordinates and color.
    Args:
        rgb (PIL.Image): RGB image to draw on.
        xy (torch.Tensor or np.ndarray): Coordinates of the lines to draw.
            Can be 1D or 2D array-like structure.
        linewidth (int): Width of the lines to draw.
        color (tuple or callable): Color of the lines. If callable, it should return a color tuple.
    
    Mode:
        ndim == 2: xy is a 2D array of shape (N, 2) where N is the number of lines.
        ndim == 3: xy is a 3D array of shape (M, N, 2) where M is the number of lines and N is the number of points per line.
    """

    draw = ImageDraw.Draw(rgb)
    if isinstance(xy, torch.Tensor):
        xy = xy.cpu().numpy()

    if xy.ndim == 2:
        if callable(color):
            cur_color = color()
        else:
            cur_color = color
        # Convert coordinates to integers and handle numpy arrays
        xy = np.round(xy).astype(int)
        for j in range(xy.shape[0]-1):
            src_xy = xy[j]
            dst_xy = xy[j + 1]
            draw.line(
                (src_xy[0], src_xy[1], dst_xy[0], dst_xy[1]),
                fill=tuple(cur_color),
                width=linewidth,
            )
    elif xy.ndim == 3:
        xy = np.round(xy).astype(int)

        for i in range(xy.shape[0]):
            if callable(color):
                cur_color = color()
            else:
                cur_color = color
            for j in range(xy.shape[1]-1):
                src_xy = xy[i, j]
                dst_xy = xy[i, j + 1]
                draw.line(
                    (src_xy[0], src_xy[1], dst_xy[0], dst_xy[1]),
                    fill=tuple(cur_color),
                    width=linewidth,
                )
    else:
        raise ValueError("xy must be 2D or 3D arrays.")
    return rgb