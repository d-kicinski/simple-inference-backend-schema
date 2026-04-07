from __future__ import annotations

from array import array
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
import threading
import time
from typing import Any, Callable, Protocol, Sequence


AudioFrame = Sequence[float] | array
ModelRunner = Callable[[array, int], Any]


class ResultSink(Protocol):
    def put(self, item: "InferenceResult") -> None:
        ...


@dataclass(slots=True)
class ModelSchedule:
    name: str
    cadence_seconds: float
    window_seconds: float
    runner: ModelRunner
    warmup_seconds: float | None = None


@dataclass(slots=True)
class InferenceResult:
    model_name: str
    trigger_sample: int
    window_start_sample: int
    window_end_sample: int
    sample_rate: int
    started_at: float
    finished_at: float
    payload: Any = None
    error: str | None = None

    @property
    def trigger_seconds(self) -> float:
        return self.trigger_sample / self.sample_rate

    @property
    def window_start_seconds(self) -> float:
        return self.window_start_sample / self.sample_rate

    @property
    def window_end_seconds(self) -> float:
        return self.window_end_sample / self.sample_rate


@dataclass(slots=True)
class _CompiledSchedule:
    spec: ModelSchedule
    cadence_samples: int
    window_samples: int
    next_trigger_sample: int


@dataclass(slots=True)
class _AudioChunk:
    start_sample: int
    samples: array

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.samples)


class RollingAudioBuffer:
    def __init__(self, max_seconds: float, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")

        self.sample_rate = sample_rate
        self.max_samples = max(1, round(max_seconds * sample_rate))
        self._chunks: deque[_AudioChunk] = deque()
        self._next_sample = 0
        self._buffered_samples = 0

    @property
    def total_samples(self) -> int:
        return self._next_sample

    @property
    def oldest_sample(self) -> int:
        if not self._chunks:
            return self._next_sample
        return self._chunks[0].start_sample

    def append(self, frame: AudioFrame) -> tuple[int, int]:
        samples = self._coerce_frame(frame)
        if not samples:
            return self._next_sample, self._next_sample

        start_sample = self._next_sample
        self._chunks.append(_AudioChunk(start_sample=start_sample, samples=samples))
        self._next_sample += len(samples)
        self._buffered_samples += len(samples)
        self._trim()
        return start_sample, self._next_sample

    def read_window(self, end_sample: int, window_samples: int) -> array:
        if window_samples <= 0:
            raise ValueError("window_samples must be > 0")

        start_sample = end_sample - window_samples
        if end_sample > self._next_sample:
            raise ValueError("requested end_sample is newer than buffered audio")
        if start_sample < self.oldest_sample:
            raise ValueError("requested window is no longer in the rolling buffer")

        window = array("f")
        remaining_start = start_sample

        for chunk in self._chunks:
            if chunk.end_sample <= remaining_start:
                continue
            if chunk.start_sample >= end_sample:
                break

            slice_start = max(0, remaining_start - chunk.start_sample)
            slice_end = min(len(chunk.samples), end_sample - chunk.start_sample)
            if slice_start < slice_end:
                window.extend(chunk.samples[slice_start:slice_end])
                remaining_start = chunk.start_sample + slice_end
            if remaining_start >= end_sample:
                break

        if len(window) != window_samples:
            raise RuntimeError(
                f"expected {window_samples} samples but collected {len(window)}"
            )

        return window

    def _trim(self) -> None:
        while self._buffered_samples > self.max_samples and self._chunks:
            overflow = self._buffered_samples - self.max_samples
            oldest = self._chunks[0]

            if len(oldest.samples) <= overflow:
                self._chunks.popleft()
                self._buffered_samples -= len(oldest.samples)
                continue

            oldest.samples = oldest.samples[overflow:]
            oldest.start_sample += overflow
            self._buffered_samples -= overflow

    @staticmethod
    def _coerce_frame(frame: AudioFrame) -> array:
        if isinstance(frame, array):
            if frame.typecode == "f":
                return array("f", frame)
            return array("f", frame.tolist())
        return array("f", frame)


class StreamingInferenceBackend:
    def __init__(
        self,
        sample_rate: int,
        schedules: Sequence[ModelSchedule],
        *,
        max_buffer_seconds: float = 30.0,
        result_sink: ResultSink | None = None,
        executor_workers: int | None = None,
    ) -> None:
        if not schedules:
            raise ValueError("at least one ModelSchedule is required")

        self.sample_rate = sample_rate
        self.buffer = RollingAudioBuffer(max_buffer_seconds, sample_rate)
        self.result_queue: Queue[InferenceResult] = Queue()
        self.result_sink = result_sink
        self._lock = threading.Lock()
        self._closed = False
        self._pending: set[Future[Any]] = set()

        self._schedules = [
            self._compile_schedule(spec)
            for spec in schedules
        ]

        largest_window = max(schedule.window_samples for schedule in self._schedules)
        if largest_window > self.buffer.max_samples:
            raise ValueError("max_buffer_seconds must be >= the largest model window")

        worker_count = executor_workers or len(self._schedules)
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="inference",
        )

    def push_audio_frame(self, frame: AudioFrame) -> None:
        with self._lock:
            self._ensure_open()
            _, stream_end = self.buffer.append(frame)
            ready_jobs: list[tuple[_CompiledSchedule, int, array]] = []

            for schedule in self._schedules:
                while stream_end >= schedule.next_trigger_sample:
                    trigger_sample = schedule.next_trigger_sample
                    window = self.buffer.read_window(
                        end_sample=trigger_sample,
                        window_samples=schedule.window_samples,
                    )
                    ready_jobs.append((schedule, trigger_sample, window))
                    schedule.next_trigger_sample += schedule.cadence_samples

        for schedule, trigger_sample, window in ready_jobs:
            future = self._executor.submit(
                self._run_inference,
                schedule.spec,
                trigger_sample,
                window,
            )
            self._track_future(future)

    def drain_results(self) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                return results

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "StreamingInferenceBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _compile_schedule(self, spec: ModelSchedule) -> _CompiledSchedule:
        cadence_samples = self._seconds_to_samples(spec.cadence_seconds)
        window_samples = self._seconds_to_samples(spec.window_seconds)
        warmup_seconds = spec.warmup_seconds if spec.warmup_seconds is not None else spec.window_seconds
        warmup_samples = self._seconds_to_samples(warmup_seconds)

        return _CompiledSchedule(
            spec=spec,
            cadence_samples=cadence_samples,
            window_samples=window_samples,
            next_trigger_sample=max(cadence_samples, warmup_samples),
        )

    def _run_inference(
        self,
        spec: ModelSchedule,
        trigger_sample: int,
        window: array,
    ) -> InferenceResult:
        started_at = time.time()
        error: str | None = None
        payload: Any = None

        try:
            payload = spec.runner(window, self.sample_rate)
        except Exception as exc:  # pragma: no cover - surfaced in result object
            error = str(exc)

        result = InferenceResult(
            model_name=spec.name,
            trigger_sample=trigger_sample,
            window_start_sample=trigger_sample - len(window),
            window_end_sample=trigger_sample,
            sample_rate=self.sample_rate,
            started_at=started_at,
            finished_at=time.time(),
            payload=payload,
            error=error,
        )
        self.result_queue.put(result)
        if self.result_sink is not None and self.result_sink is not self.result_queue:
            self.result_sink.put(result)
        return result

    def _track_future(self, future: Future[Any]) -> None:
        self._pending.add(future)

        def _cleanup(done: Future[Any]) -> None:
            self._pending.discard(done)

        future.add_done_callback(_cleanup)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("backend is closed")

    def _seconds_to_samples(self, seconds: float) -> int:
        if seconds <= 0:
            raise ValueError("schedule durations must be > 0")
        return max(1, round(seconds * self.sample_rate))
