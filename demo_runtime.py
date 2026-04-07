from __future__ import annotations

import math
from queue import Empty, Queue
import threading
import time
from typing import Any

from backend import InferenceResult, ModelSchedule, StreamingInferenceBackend


def short_window_model(samples, sample_rate: int) -> dict[str, float]:
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    return {"rms": round(rms, 6), "seconds": len(samples) / sample_rate}


def medium_window_model(samples, sample_rate: int) -> dict[str, float]:
    peak = max(abs(sample) for sample in samples)
    return {"peak": round(peak, 6), "seconds": len(samples) / sample_rate}


def long_window_model(samples, sample_rate: int) -> dict[str, float]:
    mean_abs = sum(abs(sample) for sample in samples) / len(samples)
    return {"mean_abs": round(mean_abs, 6), "seconds": len(samples) / sample_rate}


def build_demo_backend(result_sink: Queue[InferenceResult]) -> StreamingInferenceBackend:
    return StreamingInferenceBackend(
        sample_rate=16_000,
        max_buffer_seconds=30,
        result_sink=result_sink,
        schedules=[
            ModelSchedule(
                name="short-window-model",
                cadence_seconds=1,
                window_seconds=2,
                runner=short_window_model,
            ),
            ModelSchedule(
                name="medium-window-model",
                cadence_seconds=10,
                window_seconds=10,
                runner=medium_window_model,
            ),
            ModelSchedule(
                name="long-window-model",
                cadence_seconds=30,
                window_seconds=30,
                runner=long_window_model,
            ),
        ],
    )


def format_result(result: InferenceResult) -> str:
    if result.error:
        return (
            f"{result.model_name} failed at {result.trigger_seconds:.1f}s: "
            f"{result.error}"
        )
    return (
        f"{result.model_name} "
        f"window={result.window_start_seconds:.1f}-{result.window_end_seconds:.1f}s "
        f"payload={result.payload}"
    )


class DemoController:
    def __init__(
        self,
        *,
        duration_seconds: int = 35,
        frame_duration_seconds: float = 0.1,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.frame_duration_seconds = frame_duration_seconds
        self.result_queue: Queue[InferenceResult] = Queue()
        self.finished = threading.Event()
        self._stop_requested = threading.Event()
        self._producer_lock = threading.Lock()
        self._producer: threading.Thread | None = None
        self._backend: StreamingInferenceBackend | None = None

    def start(self) -> bool:
        with self._producer_lock:
            if self._producer is not None and self._producer.is_alive():
                return False

            self.result_queue = Queue()
            self.finished.clear()
            self._stop_requested.clear()
            self._backend = build_demo_backend(self.result_queue)
            self._producer = threading.Thread(
                target=self._produce_audio,
                name="audio-producer",
                daemon=True,
            )
            self._producer.start()
            return True

    def stop(self) -> None:
        self._stop_requested.set()
        backend = self._backend
        if backend is not None:
            backend.close(wait=False)

    def drain_results(self) -> list[InferenceResult]:
        results: list[InferenceResult] = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                return results

    def is_running(self) -> bool:
        producer = self._producer
        return producer is not None and producer.is_alive()

    def status_text(self, latest_by_model: dict[str, Any]) -> str:
        state = "running" if self.is_running() else "idle"
        if self.finished.is_set():
            state = "finished"
        return f"stream={state}, latest_models={len(latest_by_model)}"

    def _produce_audio(self) -> None:
        backend = self._backend
        if backend is None:
            self.finished.set()
            return

        sample_rate = backend.sample_rate
        frame_size = int(sample_rate * self.frame_duration_seconds)
        total_frames = int(self.duration_seconds / self.frame_duration_seconds)
        phase = 0

        try:
            for _ in range(total_frames):
                if self._stop_requested.is_set():
                    break
                frame = [
                    0.2 * math.sin(2 * math.pi * 220 * (phase + index) / sample_rate)
                    for index in range(frame_size)
                ]
                phase += frame_size
                backend.push_audio_frame(frame)
                time.sleep(self.frame_duration_seconds / 5)
        finally:
            backend.close()
            self.finished.set()
