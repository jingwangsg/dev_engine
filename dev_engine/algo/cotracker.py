import torch
import torchcodec
from einops import rearrange
from dev_engine.algo.faiss_kmeans import cluster_traces_kmeans
from cotracker.predictor import CoTrackerPredictor
import numpy as np
import os
import os.path as osp
from typing import List
from torchvision import transforms as T
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
import cv2
import subprocess


def decode_video(
    video_path: str, frame_size: int = 256, max_frames: int = 512
) -> Tuple[torch.Tensor, float]:
    video_dec = torchcodec.decoders.VideoDecoder(
        video_path, device="cpu", num_ffmpeg_threads=0
    )
    total_frames = video_dec.metadata.num_frames
    if total_frames <= max_frames:
        stride = 1
    else:
        stride = (total_frames + max_frames - 1) // max_frames
    frames = video_dec[::stride]
    # resize_ratio = 512 / min(frames.shape[-2:])
    frames = T.Resize(frame_size)(frames)

    return frames, total_frames


class TraceLabeling:
    def __init__(
        self, frame_size: int = 256, grid_size: int = 10, max_frames: int = 512
    ):
        # self.cotracker = torch.hub.load(
        #     "facebookresearch/co-tracker", "cotracker3_offline"
        # ).cuda()
        self.cotracker = CoTrackerPredictor(
            checkpoint="/mnt/amlfs-01/home/jingwang/PROJECTS/vla_explore/trace_prompt/data_engine_0614/checkpoints/scaled_offline.pth",
            v2=False,
            offline=True,
            window_len=60,
        ).cuda()
        self.grid_size = grid_size
        self.frame_size = frame_size
        self.max_frames = max_frames
        print("Loaded cotracker cotracker3_offline")
        self.executor = ProcessPoolExecutor(max_workers=4)

    @torch.no_grad()
    def get_point_trace(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:

        st = time.time()

        try:
            frames, total_frames = decode_video(
                video_path, frame_size=self.frame_size, max_frames=512
            )
        except Exception as e:
            print(f"Error decoding video {video_path}: {e}")
            subprocess.run(f"rm -rf {video_path}", shell=True)
            return video_path, None, None

        frames = rearrange(frames, "t c h w -> 1 t c h w").contiguous().float()
        frames = frames.cuda()
        size = frames.shape[-2:]
        print(f"Decoded video in {time.time() - st:.2f}s")

        st = time.time()
        pred_tracks, pred_visibility = self.cotracker(frames, grid_size=self.grid_size)
        print(f"Predicted in {time.time() - st:.2f}s")
        pred_tracks = pred_tracks[0]
        pred_tracks[..., 0] /= size[0]
        pred_tracks[..., 1] /= size[1]
        pred_tracks = pred_tracks.detach().cpu().numpy()
        pred_visibility = pred_visibility.detach().cpu().numpy()[0]

        pred_tracks, pred_visibility = self.interpolate_tracks(
            pred_tracks, pred_visibility, total_frames
        )

        return video_path, pred_tracks, pred_visibility

    def cotracker_inference(self, frames: torch.Tensor):
        pred_tracks, pred_visibility = self.cotracker(frames, grid_size=self.grid_size)

        return pred_tracks, pred_visibility

    def interpolate_tracks(
        self, pred_tracks: np.ndarray, pred_visibility: np.ndarray, total_frames: int
    ):
        # Get current number of frames and points
        num_frames_tracked = pred_tracks.shape[0]
        num_points = pred_tracks.shape[1]

        # Create interpolation indices
        src_indices = np.linspace(0, num_frames_tracked - 1, num_frames_tracked)
        dst_indices = np.linspace(0, num_frames_tracked - 1, total_frames)

        # Initialize interpolated arrays
        interpolated_tracks = np.zeros((total_frames, num_points, 2))
        interpolated_visibility = np.zeros((total_frames, num_points))

        # Interpolate each point trajectory
        for p in range(num_points):
            # Interpolate x and y coordinates
            interpolated_tracks[:, p, 0] = np.interp(
                dst_indices, src_indices, pred_tracks[:, p, 0]
            )
            interpolated_tracks[:, p, 1] = np.interp(
                dst_indices, src_indices, pred_tracks[:, p, 1]
            )

            # Interpolate visibility (keeping binary nature)
            # For each target frame, find the nearest source frame and use its visibility
            for i, idx in enumerate(dst_indices):
                # Find the closest source frame
                closest_idx = np.abs(src_indices - idx).argmin()
                interpolated_visibility[i, p] = pred_visibility[closest_idx, p]

        # Update with interpolated values
        pred_tracks = interpolated_tracks
        pred_visibility = interpolated_visibility

        return pred_tracks, pred_visibility

    @torch.no_grad()
    def get_point_trace_batch(
        self, video_paths: List[str]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        futures = []
        for idx, video_path in enumerate(video_paths):
            future = self.executor.submit(
                decode_video,
                video_path,
                frame_size=self.frame_size,
                max_frames=self.max_frames,
            )
            future._video_path = video_path
            futures.append(future)

        video_path_batch = []
        pred_tracks_batch = []
        pred_visibility_batch = []
        for future in as_completed(futures):
            video_path = future._video_path
            frames, total_frames = future.result()
            frames = rearrange(frames, "t c h w -> 1 t c h w").contiguous().float()
            size = frames.shape[-2:]

            frames = frames.cuda()
            print(f"Frame shape: {frames.shape}")

            pred_tracks, pred_visibility = self.cotracker_inference(frames)

            pred_tracks = pred_tracks.detach().cpu().numpy()[0]
            pred_visibility = pred_visibility.detach().cpu().numpy()[0]

            # pred_tracks is (x, y).
            pred_tracks[..., 0] /= size[1]
            pred_tracks[..., 1] /= size[0]

            # interpolate pred_tracks to original total frames
            if frames.shape[1] != total_frames:
                pred_tracks, pred_visibility = self.interpolate_tracks(
                    pred_tracks, pred_visibility, total_frames
                )

            video_path_batch.append(video_path)
            pred_tracks_batch.append(pred_tracks)
            pred_visibility_batch.append(pred_visibility)

            del frames
            torch.cuda.empty_cache()

        return video_path_batch, pred_tracks_batch, pred_visibility_batch
