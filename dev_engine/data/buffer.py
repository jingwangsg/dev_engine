# To properly support resuming, set ignore_data_skip for huggingface trainer
import os
import queue
import threading
import time
from typing import Any, Dict, Iterator, List

from dev_engine.utils.dist import get_rank, get_world_size
from loguru import logger
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset


class RandomIndicesGenerator:
    def __init__(self, n: int, seed: int = 0):
        assert n > 0, "n must be positive"
        self.n = n
        self.i = 0
        self.y = None
        self.seed = seed
        self.out_count = 0

    def __len__(self):
        return self.n

    def _ceil_log2(self) -> int:
        if self.n <= 1:
            return 0
        return (self.n - 1).bit_length()

    def _permute_k_bits(self, x: int, k: int) -> int:
        """
        A SplitMix-like bijective mixer restrained to k bits.
        Reference: https://rosettacode.org/wiki/Pseudo-random_numbers/Splitmix64
        All operations are modulo 2^k; odd multipliers keep it invertible.
        """
        mask = (1 << k) - 1
        x = (x + (self.seed & mask)) & mask

        # SplitMix64 finalizer steps, but applied under k-bit masking.
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & mask
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & mask
        x ^= x >> 31
        return x & mask

    def state_dict(self):
        return {
            "n": self.n,
            "i": self.i,
            "y": self.y,
            "seed": self.seed,
            "out_count": self.out_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.n = state_dict["n"]
        self.i = state_dict["i"]
        self.y = state_dict["y"]
        self.out_count = state_dict["out_count"]
        self.seed = state_dict["seed"]

    def __iter__(self):
        """
        Yields a permutation of range(total) in O(1) memory.
        Deterministic given `seed`. Works well up to very large totals.
        """
        if self.n < 0:
            raise ValueError("total must be non-negative")
        if self.n <= 1:
            # Trivial cases
            for i in range(self.n):
                yield i
            return

        k = self._ceil_log2()

        while self.out_count < self.n:
            self.y = self._permute_k_bits(self.i, k)
            self.i += 1
            if self.y < self.n:
                self.out_count += 1
                yield self.y


class DistributedRandomIndicesGenerator:
    """
    1. make sure each rank has the same number of buffers
    2. support distributed sampling with skip-list
    """

    def __init__(
        self,
        n: int,
        max_steps: int = 1000,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        batch_size: int = 1,
    ):
        assert rank < world_size, "rank must be less than world_size"

        self.seed = seed
        self.n = n
        self.generator = RandomIndicesGenerator(n, seed)
        self.rank = rank
        self.world_size = world_size

        self.yield_count = 0

        total_samples = max_steps * world_size * batch_size
        max_yield_per_rank = max_steps * batch_size
        max_yield_count = max_yield_per_rank * world_size
        # total samples to fetch by unified distributed buffer
        self.max_yield_count = max_yield_count

        logger.info(
            f"[Rank {rank}] DistributedRandomIndicesGenerator: total_samples: total_samples({total_samples})=max_steps({max_steps})*world_size({world_size})*batch_size({batch_size}), max_yield_count: {max_yield_count}"
        )

    def state_dict(self):
        return {
            "generator": self.generator.state_dict(),
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "yield_count": self.yield_count,
            "max_yield_count": self.max_yield_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.generator.load_state_dict(state_dict["generator"])
        self.rank = state_dict["rank"]
        self.world_size = state_dict["world_size"]
        self.seed = state_dict["seed"]
        self.yield_count = state_dict["yield_count"]
        self.max_yield_count = state_dict["max_yield_count"]

    def __len__(self):
        return self.max_yield_count

    def _get_next_index(self, it: Iterator[int]) -> tuple[int, Iterator[int]]:
        try:
            return next(it), it
        except StopIteration:
            self.seed += 1
            self.generator = RandomIndicesGenerator(self.n, self.seed)
            it = iter(self.generator)
            return next(it), it

    def __iter__(self):
        # Iterate for max_steps * batch_size times
        # Skip the initial rank indices

        # make sure when iter() is called, self.generator has already been resumed
        # Note: self.index_iterator will be reset in UnifiedDistributedBufferV2.load_state_dict()
        # reset DistributedRandomIndicesGenerator -> reset RandomIndicesGenerator
        it = iter(self.generator)
        if len(self) == 0:
            return
        for _ in range(self.rank):
            next(it)

        while self.yield_count < self.max_yield_count:
            # Only update state_dict before yield
            index, it = self._get_next_index(it)
            logger.debug(
                f"DistributedRandomIndicesGenerator: state_dict: {self.state_dict()} yield: {index}"
            )
            yield index

            self.yield_count += 1

            # Skip the next world_size - 1 indices
            for _ in range(self.world_size - 1):
                index, it = self._get_next_index(it)


class UnifiedDistributedBufferV2(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        max_steps: int = 1000,
        buffer_size: int = 1000,
        batch_size: int = 1,
        num_workers: int = 8,
        seed: int = 42,
        log_stats_interval: int = 10,
    ):
        self._detect_environment()

        self.base_dataset = base_dataset
        self.num_workers = num_workers
        self.np_rng = np.random.default_rng(seed)
        self.log_counter = 0
        self.log_stats_interval = log_stats_interval
        self.seed = seed
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.dataset_size = len(self.base_dataset)
        self.max_steps = max_steps
        self.index_generator = self._get_indices_generator()
        self.index_iterator = iter(self.index_generator)

        self.active_buffer = None
        self.active_buffer_indices = []

        # Efficient sampling state - avoids O(n*k) list reconstruction
        self.shuffled_active_indices = []
        self.current_sample_pos = 0

        # These variables should be only written by background thread
        self.fill_count = 0
        self.fill_ready_event = threading.Event()

        self._start_background_pipeline()

        # reset fill buffer and plan for what to be filled
        next_chunk_indices = self._generate_next_chunk_indices()
        self._reset_fill_buffer(next_chunk_indices)
        self._enqueue_next_chunk()

        self._closed = False

    def state_dict(self):
        # 1. update active buffer when depleted/empty
        # 2. enqueue chunk (generator state dict change)
        # 3. sample from active buffer (inside trainer loop)
        # back to 1

        # When interrupted, we will restore to the exact state at some point between Step 3
        # 1. active buffer should be refilled first (according to active_buffer_indices)
        # 2. enqueue next chunk for fill_buffer (according to fill_buffer_indices)
        # 3. index_generator should be restored to the state after generating fill_buffer_indices
        # Conclusion: save **active_buffer_indices**, **fill_buffer_indices**, **index_generator**

        return {
            "active_buffer_indices": self.active_buffer_indices,
            "fill_buffer_indices": self.fill_buffer_indices,
            "shuffled_active_indices": self.shuffled_active_indices,
            "current_sample_pos": self.current_sample_pos,
            "index_generator": self.index_generator.state_dict(),
            "np_rng_state": self.np_rng.bit_generator.state,
        }

    def _reset_fill_buffer(self, fill_buffer_indices: List[int]):
        self.fill_buffer = [None] * len(fill_buffer_indices)
        self.fill_buffer_indices = fill_buffer_indices
        self.fill_count = 0
        self.fill_ready_event.clear()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # Restart background pipeline to make sure all queues are empty
        logger.info(
            f"[Rank {self.rank}] UnifiedDistributedBufferV2: Restarting background pipeline"
        )
        logger.info(f"[Rank {self.rank}] UnifiedDistributedBufferV2: Loading state dict")
        self._stop_background_pipeline()
        self._start_background_pipeline()

        # Refill active buffer
        logger.info(f"[Rank {self.rank}] UnifiedDistributedBufferV2: Refilling active buffer")
        self.active_buffer = [None] * self.buffer_size
        self.active_buffer_indices = []
        self.shuffled_active_indices = state_dict.get("shuffled_active_indices", [])
        self.current_sample_pos = state_dict.get("current_sample_pos", 0)
        self._reset_fill_buffer(state_dict["active_buffer_indices"])
        self._enqueue_next_chunk()
        self._update_active_buffer(enqueue_next_chunk=False, reshuffle=False)

        # Enqueue fill buffer
        logger.info(f"[Rank {self.rank}] UnifiedDistributedBufferV2: Enqueuing fill buffer")
        self._reset_fill_buffer(state_dict["fill_buffer_indices"])
        self._enqueue_next_chunk()

        # Restore index generator
        self.index_generator.load_state_dict(state_dict["index_generator"])
        self.index_iterator = iter(self.index_generator)

        self.np_rng.bit_generator.state = state_dict["np_rng_state"]

    def _detect_environment(self):
        self.is_distributed = dist.is_initialized()
        self.rank = get_rank()
        self.world_size = get_world_size()

    @property
    def _fill_ready(self) -> bool:
        assert self.fill_count <= len(
            self.fill_buffer_indices
        ), f"fill_count: {self.fill_count} > fill_buffer_indices: {len(self.fill_buffer_indices)}"
        return self.fill_count == len(self.fill_buffer_indices)

    def _background_loop(self):
        try:
            while True:
                try:
                    item = self.result_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    logger.debug("Background loop: received end signal: Exiting")
                    break

                idx_in_buffer, sample, dt = item

                self.fill_buffer[idx_in_buffer] = sample
                self.fill_count += 1
                self._check_fill_count("background_loop")

                # Signal when buffer is ready
                if self._fill_ready:
                    self.fill_ready_event.set()
        except Exception as e:
            logger.error(f"Background loop error: {e}")
            raise e

    def _start_background_pipeline(self):
        self.workers = []
        self.work_queue = mp.Queue(maxsize=self.buffer_size)
        self.result_queue = mp.Queue(maxsize=self.buffer_size)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if mp.get_start_method(allow_none=True) != "fork":
            mp.set_start_method("fork", force=True)

        for wid in range(self.num_workers):
            p = mp.Process(
                target=UnifiedDistributedBufferV2._worker_process,
                args=(wid, self.base_dataset, self.work_queue, self.result_queue),
            )
            p.start()
            self.workers.append(p)
            logger.info(f"[Rank {self.rank}] UnifiedDistributedBufferV2: Started worker {wid}")

        self.background_thread = threading.Thread(target=self._background_loop, daemon=True)
        self.background_thread.start()

    def _stop_background_pipeline(self):
        # Send end signal to workers
        for _ in range(self.num_workers):
            try:
                self.work_queue.put_nowait(None)
            except (queue.Full, ValueError, OSError):
                pass

        # Send end signal to background thread
        try:
            self.result_queue.put_nowait(None)
        except (queue.Full, ValueError, OSError):
            pass

        # Wait for workers with timeout and force terminate if needed
        for p in self.workers:
            if p.is_alive():
                p.join(timeout=2.0)
                if p.is_alive():
                    logger.warning(
                        f"[Rank {self.rank}] Worker process {p.pid} did not exit cleanly, terminating"
                    )
                    p.terminate()
                    p.join(timeout=1.0)
                    if p.is_alive():
                        logger.error(
                            f"[Rank {self.rank}] Worker process {p.pid} still alive after terminate, killing"
                        )
                        p.kill()

        # Wait for background thread with timeout
        if hasattr(self, "background_thread") and self.background_thread.is_alive():
            self.background_thread.join(timeout=2.0)
            if self.background_thread.is_alive():
                logger.warning(f"[Rank {self.rank}] Background thread did not exit cleanly")

        if hasattr(self, "background_thread"):
            del self.background_thread

        self.workers.clear()

        # Close queues safely
        if hasattr(self, "work_queue"):
            try:
                self.work_queue.close()
            except (AttributeError, OSError):
                pass
            del self.work_queue

        if hasattr(self, "result_queue"):
            try:
                self.result_queue.close()
            except (AttributeError, OSError):
                pass
            del self.result_queue

    def _get_indices_generator(self):
        index_generator = DistributedRandomIndicesGenerator(
            self.dataset_size,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            seed=self.seed,
            rank=self.rank,
            world_size=self.world_size,
        )
        return index_generator

    def _check_queue_size(self, prefix):
        # logger.debug(f"[Rank {self.rank}] UnifiedDistributedBufferV2: {prefix} work_queue size: {self.work_queue.qsize()} result_queue size: {self.result_queue.qsize()}")
        pass

    def _check_fill_count(self, prefix):
        # logger.debug(f"[Rank {self.rank}] UnifiedDistributedBufferV2: {prefix} fill_count: {self.fill_count}")
        pass

    def _enqueue_next_chunk(self):
        logger.debug(f"[Rank {self.rank}] UnifiedDistributedBufferV2: Enqueuing next chunk")
        # enqueue according to fill_buffer_indices
        next_chunk_args = [
            (self.fill_buffer_indices[i], i) for i in range(len(self.fill_buffer_indices))
        ]
        for i in range(len(next_chunk_args)):
            self._check_queue_size("enqueue_next_chunk")
            self.work_queue.put_nowait(next_chunk_args[i])

    def _generate_next_chunk_indices(self):
        next_chunk_indices = []
        for _ in range(self.buffer_size):
            try:
                next_chunk_indices.append(next(self.index_iterator))
            except StopIteration:
                logger.warning(
                    f"[Rank {self.rank}] UnifiedDistributedBufferV2: index_iterator exhausted"
                )
                break
        return next_chunk_indices

    def _update_active_buffer(self, enqueue_next_chunk: bool = True, reshuffle: bool = True):
        # 1. update active buffer when fill buffer is ready
        # 2. clean up fill buffer
        # 3. enqueue next chunk
        start_time = time.time()

        # Wait efficiently using threading.Event instead of polling
        self.fill_ready_event.wait()

        logger.info(
            f"[Rank {self.rank}] UnifiedDistributedBufferV2: wait buffer ready: {time.time() - start_time} seconds"
        )
        self._check_queue_size("update_active_buffer")

        # Swap buffers atomically
        self.active_buffer = self.fill_buffer
        self.active_buffer_indices = self.fill_buffer_indices

        # Setup efficient sampling: pre-shuffle indices once (only if reshuffle=True)
        if reshuffle:
            self.shuffled_active_indices = list(range(len(self.active_buffer)))
            self.np_rng.shuffle(self.shuffled_active_indices)
            self.current_sample_pos = 0

        # Enqueue next chunk
        if enqueue_next_chunk:
            next_chunk_indices = self._generate_next_chunk_indices()

            self._reset_fill_buffer(next_chunk_indices)
            self._enqueue_next_chunk()

    @staticmethod
    def _worker_process(
        wid: int,
        base_dataset: Dataset,
        work_q: "mp.Queue[tuple[int, int]]",
        result_q: "mp.Queue[tuple[int, Any, float]]",
    ) -> None:
        torch.set_num_threads(1)
        logger.debug(f"Worker {wid} starting")
        try:
            while True:
                item = work_q.get()
                if item is None:
                    logger.debug(f"Worker {wid} received end signal: Exiting")
                    break

                idx, idx_in_buffer = item

                t0 = time.time()
                sample = base_dataset[idx]
                dt = time.time() - t0

                result_q.put((idx_in_buffer, sample, dt))
        except Exception as e:
            logger.error(f"Worker {wid} error: {e}")
            raise e

        logger.debug(f"Worker {wid} exiting cleanly")

    def _sample_active_buffer(self, batch_size: int) -> List[Any]:
        # Check if we need to refill the active buffer
        remaining_samples = len(self.shuffled_active_indices) - self.current_sample_pos
        if remaining_samples == 0:
            self._update_active_buffer()
            remaining_samples = len(self.shuffled_active_indices) - self.current_sample_pos

        # Determine how many samples we can get from current buffer
        active_to_fetch = min(batch_size, remaining_samples)

        # Get indices from pre-shuffled list (O(1) per sample)
        start_pos = self.current_sample_pos
        end_pos = start_pos + active_to_fetch
        buffer_indices_to_sample = self.shuffled_active_indices[start_pos:end_pos]

        # Update position for next sampling
        self.current_sample_pos = end_pos

        # Sample the actual data items
        sampled_items = [self.active_buffer[i] for i in buffer_indices_to_sample]

        logger.debug(
            f"[Rank {self.rank}] UnifiedDistributedBufferV2: sampled buffer indices: {buffer_indices_to_sample}"
        )
        logger.debug(
            f"[Rank {self.rank}] UnifiedDistributedBufferV2: remaining samples: {remaining_samples - active_to_fetch}"
        )

        return sampled_items

    def sample_batch(self, batch_size: int) -> List[Any]:
        batch = []

        while batch_size > 0:
            fetch_from_left = self._sample_active_buffer(batch_size)
            batch_size -= len(fetch_from_left)
            batch += fetch_from_left

        return batch

    def sample_item(self) -> Any:
        return self.sample_batch(1)[0]

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.base_dataset, name):
            return getattr(self.base_dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self) -> int:
        return self.max_steps * self.batch_size

    def __getitem__(self, idx: int) -> Any:
        return self.sample_item()

    def __del__(self):
        if not self._closed:
            self.close()

    def close(self):
        if not self._closed:
            self._stop_background_pipeline()
            self._closed = True
