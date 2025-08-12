import json
import numpy as np
import torch
import torchvision
import copy
import time
from torchcodec import VideoDecoder


def read_video_with_sampling(
    video_dec: VideoDecoder, target_fps: int = None, target_frames: int = None
):
    duration = video_dec.metadata.duration_seconds

    timestamps = None
    if target_fps is not None:
        timestamps = np.linspace(0, duration, target_fps)
    elif target_frames is not None:
        timestamps = np.arange(0, target_frames) * duration / target_frames
    else:
        raise ValueError("Either fps or num_frames must be provided")

    if timestamps is not None:
        frames = video_dec.get_frames_played_at(timestamps).data
    else:
        frames = video_dec[:]

    return frames


def smart_resize(frames, frame_size=512, divided_by_2=True, resize_when="larger_than"):
    """
    Args:
        frames (torch.Tensor): (T, C, H, W) or (C, H, W)
        frame_size (int, optional): target frame size. Defaults to 512.

    Returns:
        torch.Tensor: (T, C, H', W') or (C, H', W')
    """
    is_image = False
    if len(frames.shape) == 3:
        is_image = True
        frames = frames.unsqueeze(0)

    orig_shape = [frames.shape[-2], frames.shape[-1]]
    target_shape = copy.deepcopy(orig_shape)
    aspect_ratio = orig_shape[0] / orig_shape[1]
    if resize_when == "any" or (
        resize_when == "larger_than"
        and (orig_shape[0] > frame_size or orig_shape[1] > frame_size)
    ):
        if orig_shape[0] > orig_shape[1]:
            target_shape = [frame_size, frame_size / aspect_ratio]
        else:
            target_shape = [frame_size * aspect_ratio, frame_size]

    if divided_by_2:
        target_shape = [target_shape[0] // 2 * 2, target_shape[1] // 2 * 2]
    target_shape = [int(target_shape[0]), int(target_shape[1])]

    if target_shape != orig_shape:
        frames = torchvision.transforms.Resize(target_shape)(frames)

    if is_image:
        frames = frames.squeeze(0)

    return frames


def write_video_and_meta(video_path, frames, fps=2):
    st = time.time()
    torchvision.io.write_video(video_path, frames, fps=fps)
    print(f"Time taken for writing video: {time.time() - st}")

    st = time.time()
    video_meta_path = video_path.replace(".mp4", ".json")
    with open(video_meta_path, "w") as f:
        video_dec_dumped = VideoDecoder(video_path, device="cpu", num_ffmpeg_threads=0)
        video_meta = {
            "num_frames": video_dec_dumped.metadata.num_frames,
            "fps": video_dec_dumped.metadata.average_fps,
            "width": video_dec_dumped.metadata.width,
            "height": video_dec_dumped.metadata.height,
            "bit_rate": video_dec_dumped.metadata.bit_rate,
            "codec": video_dec_dumped.metadata.codec,
            "duration": video_dec_dumped.metadata.duration_seconds_from_header,
        }
        json.dump(video_meta, f, indent=4)
    print(f"Time taken for writing video meta: {time.time() - st}")
