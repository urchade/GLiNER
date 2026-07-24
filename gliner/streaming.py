"""Synchronous fixed-batch and asynchronous streaming inference surfaces."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from functools import partial
from threading import RLock
from dataclasses import dataclass
from collections.abc import Iterable, AsyncIterable
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    import torch

    from .model import StreamingSpanGLiNER
    from .modeling.cache import CacheState


@dataclass
class _PersistentBatchState:
    """Tensor and semantic state owned by one fixed-order streaming batch."""

    past_key_values: Any
    attention_mask: torch.Tensor
    past_word_embeddings: torch.Tensor
    past_word_mask: torch.Tensor
    prompts_embedding: torch.Tensor
    prompts_mask: torch.Tensor
    cached_lengths: torch.Tensor
    next_position_ids: torch.Tensor
    sessions: list[CacheState]

    @property
    def batch_size(self) -> int:
        return len(self.sessions)

    @property
    def physical_length(self) -> int:
        return self.attention_mask.size(1)


class StreamingBatch:
    """Persistent fixed-order batch for synchronous streaming inference.

    The session-to-row mapping is immutable, allowing the decoder KV cache to
    remain batched between calls without stacking or splitting historical keys.
    """

    def __init__(
        self,
        model: StreamingSpanGLiNER,
        session_ids: list[str],
        labels: list[str],
    ) -> None:
        normalized_ids = list(session_ids)
        if not normalized_ids or not all(isinstance(value, str) and value for value in normalized_ids):
            raise ValueError("session_ids must be a non-empty list of non-empty strings")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("session_ids must be unique within a streaming batch")
        normalized_labels = list(dict.fromkeys(labels))
        if not normalized_labels:
            raise ValueError("At least one label is required")

        self.model = model
        self.session_ids = tuple(normalized_ids)
        self.labels = tuple(normalized_labels)
        self._state: _PersistentBatchState | None = None
        self._closed = False
        self._lock = RLock()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def initialized(self) -> bool:
        return self._state is not None

    def append(
        self,
        texts: list[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
        recompute: bool = False,
    ) -> list[list[dict[str, Any]]]:
        """Append one text chunk per fixed session row and return snapshots."""
        with self._lock:
            if self._closed:
                raise RuntimeError("StreamingBatch is closed")
            chunks = list(texts)
            if len(chunks) != len(self.session_ids):
                raise ValueError("texts must contain one chunk per session row")
            if not all(isinstance(text, str) for text in chunks):
                raise TypeError("Every streaming chunk must be a string")
            try:
                self._state, outputs = self.model._run_persistent_stream_batch(
                    self._state,
                    self.session_ids,
                    self.labels,
                    chunks,
                    threshold=threshold,
                    flat_ner=flat_ner,
                    multi_label=multi_label,
                    return_class_probs=return_class_probs,
                    recompute=recompute,
                )
            except BaseException:
                # Persistent KV layers append in place. Conservatively discard
                # the group if a forward or commit fails part-way through.
                self._state = None
                raise
            return outputs

    def reset(self) -> None:
        """Discard the entire persistent cache while retaining row ownership."""
        with self._lock:
            if self._closed:
                raise RuntimeError("StreamingBatch is closed")
            self._state = None

    def close(self) -> None:
        """Release the persistent batch cache. Idempotent."""
        with self._lock:
            self._state = None
            self._closed = True

    def __enter__(self) -> StreamingBatch:
        if self._closed:
            raise RuntimeError("StreamingBatch is closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


@dataclass
class _AsyncStreamRequest:
    text: str
    labels: list[str]
    session_id: str
    recompute: bool
    threshold: float
    flat_ner: bool
    multi_label: bool
    return_class_probs: bool
    future: asyncio.Future

    def as_model_request(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "labels": self.labels,
            "session_id": self.session_id,
            "recompute": self.recompute,
            "threshold": self.threshold,
            "flat_ner": self.flat_ner,
            "multi_label": self.multi_label,
            "return_class_probs": self.return_class_probs,
        }


class AsyncStreamingEngine:
    """Dynamically microbatch independently arriving streaming sessions."""

    _STOP = object()

    def __init__(
        self,
        model: StreamingSpanGLiNER,
        *,
        max_batch_size: int = 32,
        batch_wait_timeout_ms: float = 2.0,
        queue_capacity: int = 4096,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if batch_wait_timeout_ms < 0:
            raise ValueError("batch_wait_timeout_ms must be non-negative")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")

        self.model = model
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_timeout_s = float(batch_wait_timeout_ms) / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_capacity)
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._worker_task: asyncio.Task | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing = False
        self._closed = False

    async def start(self) -> AsyncStreamingEngine:
        """Start the scheduler lazily on the current event loop."""
        if self._closed:
            raise RuntimeError("AsyncStreamingEngine is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("AsyncStreamingEngine cannot move between event loops")
        self._loop = loop
        if self._worker_task is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="gliner-streaming",
            )
            self._worker_task = loop.create_task(self._worker(), name="gliner-streaming-scheduler")
        return self

    async def append(
        self,
        session_id: str,
        text: str,
        labels: list[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        return_class_probs: bool = False,
        recompute: bool = False,
    ) -> list[dict[str, Any]]:
        """Queue one append; concurrent callers are dynamically microbatched."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized_labels = list(dict.fromkeys(labels))
        if not normalized_labels:
            raise ValueError("At least one label is required")
        if self._closing or self._closed:
            raise RuntimeError("AsyncStreamingEngine is closing or closed")
        await self.start()

        # Match the synchronous compatibility API: blank appends do not mutate
        # session state and do not consume a GPU batch slot.
        if not text.strip():
            return []

        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        await lock.acquire()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = _AsyncStreamRequest(
            text=text,
            labels=normalized_labels,
            session_id=session_id,
            recompute=bool(recompute),
            threshold=float(threshold),
            flat_ner=bool(flat_ner),
            multi_label=bool(multi_label),
            return_class_probs=bool(return_class_probs),
            future=future,
        )
        try:
            await self._queue.put(request)
        except BaseException:
            lock.release()
            raise

        def release_session_lock(_future) -> None:
            if lock.locked():
                lock.release()

        future.add_done_callback(release_session_lock)
        return await asyncio.shield(future)

    async def stream(
        self,
        session_id: str,
        chunks: AsyncIterable[str] | Iterable[str],
        labels: list[str],
        **kwargs,
    ):
        """Yield a session snapshot for every chunk from sync or async input."""
        if isinstance(chunks, AsyncIterable):
            async for chunk in chunks:
                yield await self.append(session_id, chunk, labels, **kwargs)
            return
        for chunk in chunks:
            yield await self.append(session_id, chunk, labels, **kwargs)

    async def clear_session(self, session_id: str) -> None:
        """Wait for prior work on a session, then discard its cache."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            self.model.clear_session(session_id)

    async def _collect_batch(self, first: _AsyncStreamRequest) -> list[_AsyncStreamRequest]:
        batch = [first]
        if self.batch_wait_timeout_s == 0:
            while len(batch) < self.max_batch_size:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is self._STOP:
                    self._queue.task_done()
                    self._closing = True
                    break
                batch.append(item)
            return batch

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.batch_wait_timeout_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            if item is self._STOP:
                self._queue.task_done()
                self._closing = True
                break
            batch.append(item)
        return batch

    async def _worker(self) -> None:
        assert self._loop is not None
        assert self._executor is not None
        while True:
            item = await self._queue.get()
            if item is self._STOP:
                self._queue.task_done()
                return
            batch = await self._collect_batch(item)
            requests = [request.as_model_request() for request in batch]
            try:
                # One model worker owns all mutable session caches. The forward
                # itself is batched (GPU-parallel); serial dispatch avoids cache
                # races and competing CUDA launches on a single model replica.
                job = self._loop.run_in_executor(
                    self._executor,
                    partial(
                        self.model._run_session_items_batched,
                        requests,
                        return_exceptions=True,
                    ),
                )
                # Polling keeps the event loop responsive on runtimes where a
                # PyTorch worker's cross-thread wakeup can be delayed until the
                # selector's next timer event.
                while not job.done():
                    await asyncio.sleep(0.001)
                outputs = job.result()
            except BaseException as error:
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(error)
            else:
                for request, output in zip(batch, outputs):
                    if not request.future.done():
                        if isinstance(output, Exception):
                            request.future.set_exception(output)
                        else:
                            request.future.set_result(output)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def close(self) -> None:
        """Drain queued work and stop the scheduler. Idempotent."""
        if self._closed:
            return
        self._closing = True
        if self._worker_task is not None:
            await self._queue.join()
            await self._queue.put(self._STOP)
            await self._worker_task
            self._worker_task = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        self._closed = True

    async def __aenter__(self) -> AsyncStreamingEngine:
        return await self.start()

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        await self.close()


__all__ = ["AsyncStreamingEngine", "StreamingBatch"]
