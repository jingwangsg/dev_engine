#!/usr/bin/env python
"""
Unified Distributed Buffer with partitioned double-buffer swapping.

- Deterministic coverage: no overlap within a chunk.
- Last chunk padded by wrapping.
- New epoch = reshuffle + repartition.
"""
import logging
import os
import queue
import random
import threading
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset


class UnifiedDistributedBufferV2(Dataset):
    """
    Unified Distributed Buffer with partitioned double-buffer swapping.

    Features:
    - Deterministic coverage: no overlap within a chunk
    - Last chunk padded by wrapping
    - New epoch = reshuffle + repartition
    - Proper resource cleanup to prevent semaphore leaks

    Example usage:
        ```python
        # Using context manager (recommended)
        with UnifiedDistributedBufferV2(
            base_dataset=my_dataset,
            buffer_size=1000,
            num_workers=8
        ) as buffer:
            for epoch in range(num_epochs):
                for batch in DataLoader(buffer, batch_size=32):
                    # training code here
                    pass

        # Manual cleanup (if context manager not used)
        buffer = UnifiedDistributedBufferV2(my_dataset, buffer_size=1000)
        try:
            # use buffer
            pass
        finally:
            buffer.close()  # Important: prevents resource leaks
        ```
    """

    def __init__(
        self,
        base_dataset: Dataset,
        buffer_size: int = 1000,
        num_workers: int = 8,
        seed: int = 42,
        log_stats_interval: int = 10,
    ):
        # --- core params ---
        self.base_dataset = base_dataset
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.rng = random.Random(seed)
        self.shard_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.log_stats_int = log_stats_interval
        self.counter = 0
        self.depletion_count = 0

        # --- distributed / shard setup ---
        self._detect_environment()
        self._setup_sharding()

        # --- partition into fixed-size chunks ---
        self._partition_shard()
        self.current_chunk = 0

        # --- double buffers ---
        self.buffer_A: List[Optional[Any]] = [None] * buffer_size
        self.buffer_B: List[Optional[Any]] = [None] * buffer_size
        self.active_buffer = self.buffer_A
        self.fill_buffer = self.buffer_B

        # --- fill & swap state ---
        self.fill_count = 0
        self.fill_pos = 0
        self.swap_ready = threading.Event()

        # --- sampling state ---
        self.active_indices = []
        self.batches_since_swap = 0
        self.batches_before_swap_history = []
        self.batch_size = 1

        # --- locks & threads ---
        # ensure `spawn` start-method once before workers start
        # if mp.get_start_method(allow_none=True) != "spawn":
        #     mp.set_start_method("spawn", force=True)
        mp.set_start_method("fork", force=True)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        # multiprocessing queues for work and results
        self.work_queue = mp.Queue(maxsize=buffer_size * 2)
        self.result_queue = mp.Queue(maxsize=buffer_size * 2)
        # Don't remove the 2 here! Otherwise throttling will stall updates!
        self.workers: List[mp.Process] = []
        self.collector_thread: Optional[threading.Thread] = None

        if dist.is_initialized() and dist.get_world_size() > 1:
            # A lightweight CPU/Gloo group for metadata collectives
            self.ctrl_pg = dist.new_group(backend="gloo")
        else:
            self.ctrl_pg = None

        # --- stats ---
        self.stats = {
            "samples_processed": 0,
            "total_load_time": 0.0,
            "buffer_hits": 0,
            "access_count": 0,
            "lock_hold_time": 0.0,
        }
        self.stats_lock = threading.Lock()

        # --- cleanup state ---
        self._closed = False

        # --- start loaders, enqueue first chunk, do initial swap ---
        self._start_workers()
        self._enqueue_current_chunk()
        self._initial_fill_and_swap()

        print(
            f"[Rank {self.rank}] Buffer initialized: "
            f"{self.num_chunks} chunks × {buffer_size} samples each."
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def _detect_environment(self):
        if dist.is_initialized():
            self.is_distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        elif "RANK" in os.environ:
            self.is_distributed = True
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        else:
            self.is_distributed = False
            self.rank = 0
            self.world_size = 1

    def _setup_sharding(self):
        total = len(self.base_dataset)  # type: ignore
        indices = list(range(total))
        # shuffle indices
        self.shard_rng.shuffle(indices)
        if self.is_distributed and self.world_size > 1:
            per = total // self.world_size
            if per < self.buffer_size:
                print(
                    f"Rank {self.rank} has {per} samples, which is less than buffer size {self.buffer_size}. Decreasing buffer size to {per}."
                )
                self.buffer_size = per
            start = self.rank * per
            end = start + per + (total % self.world_size if self.rank == self.world_size - 1 else 0)
        else:
            start, end = 0, total
        self.shard_indices = indices[start:end]

    def _partition_shard(self):
        """Shuffle shard once per epoch, then cut into fixed-size chunks."""
        self.rng.shuffle(self.shard_indices)
        # chunk
        chunks = [
            self.shard_indices[i : i + self.buffer_size]
            for i in range(0, len(self.shard_indices), self.buffer_size)
        ]
        # pad last chunk by wrapping
        last = chunks[-1]
        if len(last) < self.buffer_size:
            wrap = self.shard_indices[: (self.buffer_size - len(last))]
            chunks[-1] = last + wrap
        self.buffer_chunks = chunks
        self.num_chunks = len(chunks)

    def _enqueue_current_chunk(self):
        """Push exactly one chunk's indices into the work-queue."""
        for pos, idx in enumerate(self.buffer_chunks[self.current_chunk]):
            try:
                self.work_queue.put_nowait((pos, idx))
            except queue.Full:
                break

    def _initial_fill_and_swap(self):
        """Wait for fill_buffer to reach buffer_size, then swap it in."""
        t0 = time.time()
        while True:
            with self.lock:
                if self.fill_count >= self.buffer_size:
                    break
            if self.fill_count == self.buffer_size // 2:
                print(
                    f"[Rank {self.rank}] Initial fill: {self.fill_count}/{self.buffer_size} takes {time.time() - t0:.1f}s"
                )
            time.sleep(0.1)
        print(
            f"[Rank {self.rank}] Initial fill: {self.fill_count}/{self.buffer_size} takes {time.time() - t0:.1f}s"
        )
        self._do_swap()

    def _global_swap_if_needed(self) -> bool:
        """
        Coordinate a *simultaneous* swap across all ranks.
        Returns True if a swap was performed by this call, else False.
        """
        if self.ctrl_pg is None:
            return False  # single‑GPU or dist not initialised

        # 1) vote: 1 = this rank is ready, 0 = not ready  (CPU tensor!)
        ready = torch.tensor(
            [1 if self.swap_ready.is_set() else 0], dtype=torch.uint8, device="cpu"
        )
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=self.ctrl_pg)

        if ready.item() == 1:
            dist.barrier(group=self.ctrl_pg)  # also on CPU/Gloo
            self._do_swap()  # <‑ no barrier inside
            return True
        return False

    def _do_swap(self):
        """Atomically swap active⇄fill, record stats, and enqueue next chunk (or reshuffle if epoch ended)."""
        print(f"[Rank {self.rank}] Swapping buffers")
        with self.lock:
            # record how many batches we served
            self.batches_before_swap_history.append(self.batches_since_swap)
            self.batches_since_swap = 0

            # swap buffers
            self.active_buffer, self.fill_buffer = self.fill_buffer, self.active_buffer

            # reset fill_buffer
            self.fill_buffer[:] = [None] * self.buffer_size
            self.fill_count = 0
            self.fill_pos = 0
            self.swap_ready.clear()

            # advance chunk pointer
            self.current_chunk += 1
            if self.current_chunk >= self.num_chunks:
                # new epoch: reshuffle & repartition
                self._partition_shard()
                self.current_chunk = 0

        # reset sample-without-replacement indices
        self.active_indices = list(range(self.buffer_size))
        self.depletion_count = 0

        # immediately enqueue the next chunk
        self._enqueue_current_chunk()

    @staticmethod
    def _worker_process(
        wid: int,
        base_dataset: Dataset,
        work_q: "mp.Queue[Any]",
        result_q: "mp.Queue[Any]",
    ) -> None:
        """Independent process executing CPU-bound dataset indexing."""
        torch.set_num_threads(1)
        logging.debug(f"Worker {wid} starting")

        while True:
            payload_in = work_q.get()
            if payload_in is None:  # shutdown signal
                break

            pos, idx = payload_in

            t0 = time.time()
            sample = base_dataset[idx]
            dt = time.time() - t0

            # note: sending (pos, sample, dt) back — collector updates stats
            result_q.put((pos, sample, dt))

        logging.debug(f"Worker {wid} exiting cleanly")

    def _collector_loop(self) -> None:
        """Thread in main process that drains result_q and fills the buffer."""
        while not self.stop_event.is_set():
            try:
                payload = self.result_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if payload is None:
                break  # explicit shutdown sentinel

            pos, sample, dt = payload

            start_lock_hold = time.time()
            with self.lock:
                if self.stop_event.is_set():
                    break

                # place sample in its expected position if not already filled
                if 0 <= pos < self.buffer_size and self.fill_buffer[pos] is None:
                    self.fill_buffer[pos] = sample
                    self.fill_count += 1
                    if self.fill_count >= self.buffer_size:
                        self.swap_ready.set()
            end_lock_hold = time.time()

            with self.stats_lock:
                self.stats["samples_processed"] += 1
                self.stats["total_load_time"] += dt
                self.stats["lock_hold_time"] += end_lock_hold - start_lock_hold

    def _start_workers(self):
        """Launch multiprocessing workers plus a single collector thread."""
        for wid in range(self.num_workers):
            p = mp.Process(
                target=UnifiedDistributedBufferV2._worker_process,
                args=(wid, self.base_dataset, self.work_queue, self.result_queue),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        # start collector thread (runs in main process)
        self.collector_thread = threading.Thread(target=self._collector_loop, daemon=True)
        self.collector_thread.start()

    def __len__(self) -> int:
        return len(self.shard_indices)

    def __getitem__(self, idx: int) -> Any:
        """Random-access from active buffer (with replacement)."""
        if self._closed:
            raise RuntimeError("Cannot access closed buffer")

        with self.stats_lock:
            self.stats["access_count"] += 1

        # if fill buffer is ready, swap it in before serving
        if self.swap_ready.is_set():
            self._do_swap()

        # Filter out None values from active buffer
        available_samples = [s for s in self.active_buffer if s is not None]
        if not available_samples:
            raise RuntimeError("No samples available in active buffer")

        sample = self.rng.choice(available_samples)
        with self.stats_lock:
            self.stats["buffer_hits"] += 1
        return sample

    def sample_batch(self, batch_size: int) -> List[Any]:
        # note that buffer size % batch size should be 0
        """Draw without replacement from active buffer, resetting when exhausted."""
        if self._closed:
            raise RuntimeError("Cannot access closed buffer")

        self.batch_size = batch_size
        self.counter += 1
        if self.counter % self.log_stats_int == 0:
            print(f"[Rank {self.rank}] statistics: {self.get_statistics()}")

        self.stats["access_count"] += batch_size

        if self.depletion_count > 0 and self.swap_ready.is_set():
            self._global_swap_if_needed()

        if len(self.active_indices) < batch_size:
            print(
                f"[Rank {self.rank}] active_indices: {len(self.active_indices)} < batch_size: {batch_size}"
            )
            self.active_indices = list(range(self.buffer_size))
            self.depletion_count += 1
            performed_swap = self._global_swap_if_needed()
            if not performed_swap:
                print(f"[Rank {self.rank}] depletion_count: {self.depletion_count}")

        # normal sampling no lock
        indices = self.np_rng.choice(self.active_indices, size=batch_size, replace=False)
        batch = [self.active_buffer[i] for i in indices]
        self.active_indices = [i for i in self.active_indices if i not in indices]

        self.batches_since_swap += 1
        self.stats["buffer_hits"] += batch_size
        return batch

    def get_statistics(self) -> Dict[str, Any]:
        with self.stats_lock:
            s = dict(self.stats)
        s["avg_load_time"] = (
            s["total_load_time"] / s["samples_processed"] if s["samples_processed"] else 0.0
        )
        s["avg_lock_hold_time"] = (
            s["lock_hold_time"] / s["samples_processed"] if s["samples_processed"] else 0.0
        )
        s["current_chunk"] = self.current_chunk
        s["num_chunks"] = self.num_chunks
        s["fill_buffer_size"] = self.fill_count
        s["count_times_per_swap"] = self.batches_since_swap * self.batch_size / self.buffer_size
        s["depletion_count"] = self.depletion_count
        return s

    def close(self):
        """Properly clean up all resources to prevent leaks."""
        if hasattr(self, "_closed") and self._closed:
            return

        logging.debug(f"[Rank {self.rank}] Starting buffer cleanup...")

        # Signal global shutdown
        self.stop_event.set()

        # Drain any pending items (best-effort)
        try:
            while not self.work_queue.empty():
                self.work_queue.get_nowait()
        except queue.Empty:
            pass

        # Poison-pill each worker
        for _ in range(self.num_workers):
            try:
                self.work_queue.put_nowait(None)
            except queue.Full:
                pass

        # Wake collector
        try:
            self.result_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for worker processes
        for i, p in enumerate(self.workers):
            if p.is_alive():
                p.join(timeout=2.0)
                if p.is_alive():
                    logging.warning(f"[Rank {self.rank}] Worker {i} did not exit cleanly")

        # Wait for collector thread
        if self.collector_thread is not None and self.collector_thread.is_alive():
            self.collector_thread.join(timeout=2.0)

        # Clear worker list
        self.workers.clear()

        # Close queues & clear events / buffers
        self.work_queue.close()
        self.result_queue.close()

        # Clear events and buffers
        if hasattr(self, "swap_ready"):
            self.swap_ready.clear()
            del self.swap_ready

        if hasattr(self, "stop_event"):
            self.stop_event.clear()
            del self.stop_event

        # Clear buffers
        self.buffer_A.clear()
        self.buffer_B.clear()

        self._closed = True
        logging.info(f"[Rank {self.rank}] Buffer closed successfully")

    def __del__(self):
        """Ensure cleanup happens even if close() wasn't called explicitly."""
        try:
            if hasattr(self, "_closed") and not self._closed:
                self.close()
        except Exception as e:
            # Avoid raising exceptions in __del__
            logging.warning(f"Error during buffer cleanup in __del__: {e}")
            pass

    def get_pad_token_id(self) -> int:
        return self.base_dataset.get_pad_token_id()  # type: ignore

    def get_padding_side(self) -> str:
        return self.base_dataset.get_padding_side()  # type: ignore

class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


