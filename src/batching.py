import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, List, Optional

logger = logging.getLogger(__name__)



# Data structures

@dataclass
class PendingRequest:
    prompt: str
    max_tokens: int
    temperature: float
    future: asyncio.Future          # Resolved with the inference result
    enqueued_at: float = field(default_factory=time.time)



# Dynamic batcher


class DynamicBatcher:
    """
    Groups concurrent inference requests into batches to amortize GPU
    weight-loading costs.

    Usage::

        batcher = DynamicBatcher(
            inference_fn=my_model.generate_batch,
            max_batch_size=8,
            batch_timeout_ms=50.0,
        )
        async with asyncio.TaskGroup() as tg:
            tg.create_task(batcher.start())
            result = await batcher.submit("Hello world", max_tokens=100)
    """

    def __init__(
        self,
        inference_fn: Callable[[List[str], List[int], float], Awaitable[List[str]]],
        max_batch_size: int = 8,
        batch_timeout_ms: float = 50.0,
    ):
        """
        Args:
            inference_fn: Async callable that accepts
                          (prompts, max_tokens_list, temperature) → [responses].
            max_batch_size: Fire batch when this many requests are pending.
            batch_timeout_ms: Fire batch after this many ms regardless of size.
        """
        self._inference_fn = inference_fn
        self._max_batch_size = max_batch_size
        self._timeout_s = batch_timeout_ms / 1000.0

        self._pending: List[PendingRequest] = []
        self._lock = asyncio.Lock()          # Protects _pending
        self._not_empty = asyncio.Event()    # Signals timeout loop

        self._running = False
        self._background_task: Optional[asyncio.Task] = None

        # Metrics
        self._total_requests = 0
        self._total_batches = 0
        self._total_batch_size_sum = 0

    #Lifecycle 
    async def start(self) -> None:
        """Start the background timeout processor.  Call once at server startup."""
        self._running = True
        self._background_task = asyncio.create_task(
            self._timeout_loop(), name="batcher-timeout-loop"
        )
        logger.info(
            "DynamicBatcher started (max_batch=%d, timeout=%.0fms)",
            self._max_batch_size,
            self._timeout_s * 1000,
        )

    async def stop(self) -> None:
        """Gracefully stop the background task."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("DynamicBatcher stopped")

    #Public API 
    async def submit(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        """
        Submit a single request and wait for its result.

        The request will be grouped with others that arrive within the
        batch window.  Returns the generated text string.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        request = PendingRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            future=future,
        )

        should_process_now = False
        async with self._lock:
            self._pending.append(request)
            self._total_requests += 1
            if len(self._pending) >= self._max_batch_size:
                should_process_now = True
            else:
                self._not_empty.set()   # Wake timeout loop if sleeping

        if should_process_now:
            asyncio.create_task(self._flush())

        return await future

    #Internal batch processing 

    async def _timeout_loop(self) -> None:
        """Background coroutine: fires partial batches when timeout elapses."""
        while self._running:
            # Wait until at least one request is pending
            await self._not_empty.wait()
            self._not_empty.clear()

            # Sleep for the configured timeout window
            await asyncio.sleep(self._timeout_s)

            # After sleeping, flush whatever is pending
            async with self._lock:
                has_pending = len(self._pending) > 0

            if has_pending:
                await self._flush()

    async def _flush(self) -> None:
        """Drain up to max_batch_size pending requests and run inference."""
        async with self._lock:
            if not self._pending:
                return
            batch = self._pending[: self._max_batch_size]
            self._pending = self._pending[self._max_batch_size :]
            # If more remain, ensure the timeout loop wakes again
            if self._pending:
                self._not_empty.set()

        batch_size = len(batch)
        self._total_batches += 1
        self._total_batch_size_sum += batch_size

        logger.debug(
            "Processing batch of %d request(s) [batch #%d]",
            batch_size,
            self._total_batches,
        )

        prompts = [r.prompt for r in batch]
        max_tokens_list = [r.max_tokens for r in batch]
        # Use the temperature from the first request (all should match for
        # cached requests; mixed-temp batches are an advanced topic).
        temperature = batch[0].temperature

        try:
            results: List[str] = await self._inference_fn(
                prompts, max_tokens_list, temperature
            )
            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batch inference failed: %s", exc)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)

    #Metrics 
    def stats(self) -> dict:
        avg_batch = (
            round(self._total_batch_size_sum / self._total_batches, 2)
            if self._total_batches
            else 0
        )
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": avg_batch,
            "max_batch_size": self._max_batch_size,
            "batch_timeout_ms": self._timeout_s * 1000,
            "pending_requests": len(self._pending),
        }