import queue
import threading


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False
        self._background_started = False

    def _start_background_pipeline_if_not_started(self):
        if self._background_started:
            return
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()
        self._background_started = True

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put((batch, self.buffer.state_dict()))

    def __iter__(self):
        # don't start background filling until it's used
        # this is to coordinate with buffer resuming

        self._start_background_pipeline_if_not_started()
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch, state_dict = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch