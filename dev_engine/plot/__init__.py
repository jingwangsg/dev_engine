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

def draw_line(rgb, src_xy, dst_xy, linewidth, color=get_rainbow_color):
    if callable(color):
        color = color()

    draw = ImageDraw.Draw(rgb)
    if isinstance(src_xy, torch.Tensor):
        src_xy = src_xy.cpu().numpy()
    if isinstance(dst_xy, torch.Tensor):
        dst_xy = dst_xy.cpu().numpy()

    # Convert coordinates to integers and handle numpy arrays
    src_xy = np.round(src_xy).astype(int)
    dst_xy = np.round(dst_xy).astype(int)
    draw.line(
        (src_xy[0], src_xy[1], dst_xy[0], dst_xy[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb