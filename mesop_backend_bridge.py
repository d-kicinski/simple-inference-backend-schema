from __future__ import annotations

from array import array
import atexit
import base64
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.queues import Queue as MpQueue
import queue
import sys
import threading
import time
from typing import Any, Literal

from backend import InferenceResult, ModelSchedule, StreamingInferenceBackend


EXPECTED_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class BackendCommand:
    kind: Literal["open_session", "audio_chunk", "close_session", "shutdown"]
    session_id: str
    pcm_s16le: bytes = b""
    sample_rate: int = EXPECTED_SAMPLE_RATE
    channels: int = 1
    created_at: float = 0.0


@dataclass(slots=True)
class BackendEnvelope:
    session_id: str
    kind: Literal["status", "plot", "text", "log"]
    payload: dict[str, Any]
    created_at: float


def short_window_model(samples: array, sample_rate: int) -> dict[str, float]:
    energy = sum(sample * sample for sample in samples) / max(1, len(samples))
    score = energy ** 0.5
    return {
        "score": round(score, 6),
        "seconds": round(len(samples) / sample_rate, 3),
    }


def medium_window_model(samples: array, sample_rate: int) -> dict[str, float]:
    peak = max(abs(sample) for sample in samples) if samples else 0.0
    return {
        "peak": round(peak, 6),
        "seconds": round(len(samples) / sample_rate, 3),
    }


def long_window_model(samples: array, sample_rate: int) -> dict[str, float]:
    mean_abs = sum(abs(sample) for sample in samples) / max(1, len(samples))
    return {
        "mean_abs": round(mean_abs, 6),
        "seconds": round(len(samples) / sample_rate, 3),
    }


class SessionRuntime:
    def __init__(self, session_id: str, output_queue: MpQueue) -> None:
        self.session_id = session_id
        self.output_queue = output_queue
        self.inference_queue: queue.Queue[InferenceResult] = queue.Queue()
        self.score_history: deque[tuple[float, float]] = deque(maxlen=120)
        self._closed = threading.Event()
        self._artifacts = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"artifacts-{session_id[:8]}",
        )
        self.backend = StreamingInferenceBackend(
            sample_rate=EXPECTED_SAMPLE_RATE,
            max_buffer_seconds=30,
            result_sink=self.inference_queue,
            executor_workers=3,
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
        self._consumer = threading.Thread(
            target=self._consume_results,
            name=f"results-{session_id[:8]}",
            daemon=True,
        )
        self._consumer.start()
        self._emit(
            "status",
            {"message": "session-opened", "sample_rate": EXPECTED_SAMPLE_RATE},
        )

    def push_audio_chunk(self, pcm_s16le: bytes, sample_rate: int, channels: int) -> None:
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"expected sample_rate={EXPECTED_SAMPLE_RATE}, got {sample_rate}"
            )
        if channels != 1:
            raise ValueError("PoC expects mono audio")
        if not pcm_s16le:
            return

        samples = decode_pcm_s16le_to_f32(pcm_s16le)
        self.backend.push_audio_frame(samples)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.backend.close()
        self._consumer.join(timeout=1)
        self._artifacts.shutdown(wait=False)
        self._emit("status", {"message": "session-closed"})

    def _consume_results(self) -> None:
        while not self._closed.is_set() or not self.inference_queue.empty():
            try:
                result = self.inference_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._emit(
                "log",
                {
                    "message": format_inference_log(result),
                    "model_name": result.model_name,
                    "trigger_seconds": round(result.trigger_seconds, 3),
                },
            )

            if result.error:
                continue

            if result.model_name == "short-window-model":
                score = float(result.payload["score"])
                self.score_history.append((result.window_end_seconds, score))
                self._artifacts.submit(
                    self._emit_plot,
                    list(self.score_history),
                    float(result.window_end_seconds),
                )
            elif result.model_name == "long-window-model":
                self._artifacts.submit(self._emit_text, dict(result.payload), result)

    def _emit_plot(
        self,
        history: list[tuple[float, float]],
        trigger_seconds: float,
    ) -> None:
        self._emit(
            "plot",
            {
                "trigger_seconds": round(trigger_seconds, 3),
                "image_url": render_score_plot(history),
                "point_count": len(history),
            },
        )

    def _emit_text(self, payload: dict[str, float], result: InferenceResult) -> None:
        summary = (
            f"30s summary at {result.window_end_seconds:.1f}s: "
            f"mean_abs={payload['mean_abs']:.4f}. "
            f"Simulated LLM says the signal looks stable."
        )
        self._emit(
            "text",
            {
                "trigger_seconds": round(result.trigger_seconds, 3),
                "summary": summary,
            },
        )

    def _emit(self, kind: Literal["status", "plot", "text", "log"], payload: dict[str, Any]) -> None:
        self.output_queue.put(
            BackendEnvelope(
                session_id=self.session_id,
                kind=kind,
                payload=payload,
                created_at=time.time(),
            )
        )


def backend_process_main(
    command_queue: MpQueue,
    result_queue: MpQueue,
) -> None:
    sessions: dict[str, SessionRuntime] = {}

    while True:
        command: BackendCommand = command_queue.get()

        if command.kind == "shutdown":
            break

        if command.kind == "open_session":
            if command.session_id not in sessions:
                sessions[command.session_id] = SessionRuntime(
                    session_id=command.session_id,
                    output_queue=result_queue,
                )
            continue

        if command.kind == "close_session":
            session = sessions.pop(command.session_id, None)
            if session is not None:
                session.close()
            continue

        if command.kind == "audio_chunk":
            session = sessions.get(command.session_id)
            if session is None:
                session = SessionRuntime(
                    session_id=command.session_id,
                    output_queue=result_queue,
                )
                sessions[command.session_id] = session
            try:
                session.push_audio_chunk(
                    pcm_s16le=command.pcm_s16le,
                    sample_rate=command.sample_rate,
                    channels=command.channels,
                )
            except Exception as exc:
                result_queue.put(
                    BackendEnvelope(
                        session_id=command.session_id,
                        kind="log",
                        payload={"message": f"audio chunk rejected: {exc}"},
                        created_at=time.time(),
                    )
                )

    for session in sessions.values():
        session.close()


class BackendBridge:
    def __init__(self, *, queue_size: int = 256) -> None:
        self._ctx = mp.get_context("spawn")
        self._queue_size = queue_size
        self._commands: MpQueue | None = None
        self._results: MpQueue | None = None
        self._process: mp.Process | None = None
        self._dispatcher: threading.Thread | None = None
        self._stop_dispatcher = threading.Event()
        self._lock = threading.Lock()
        self._session_results: dict[str, deque[BackendEnvelope]] = defaultdict(deque)

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return

            self._commands = self._ctx.Queue(maxsize=self._queue_size)
            self._results = self._ctx.Queue()
            self._stop_dispatcher.clear()
            self._process = self._ctx.Process(
                target=backend_process_main,
                args=(self._commands, self._results),
                daemon=True,
                name="mesop-audio-backend",
            )
            self._process.start()
            self._dispatcher = threading.Thread(
                target=self._dispatch_results,
                name="mesop-result-dispatcher",
                daemon=True,
            )
            self._dispatcher.start()

    def open_session(self, session_id: str) -> None:
        self.start()
        self._put_command(
            BackendCommand(
                kind="open_session",
                session_id=session_id,
                created_at=time.time(),
            )
        )

    def push_audio_chunk(
        self,
        *,
        session_id: str,
        pcm_s16le: bytes,
        sample_rate: int = EXPECTED_SAMPLE_RATE,
        channels: int = 1,
    ) -> bool:
        self.start()
        if self._commands is None:
            return False
        try:
            self._commands.put_nowait(
                BackendCommand(
                    kind="audio_chunk",
                    session_id=session_id,
                    pcm_s16le=pcm_s16le,
                    sample_rate=sample_rate,
                    channels=channels,
                    created_at=time.time(),
                )
            )
            return True
        except queue.Full:
            with self._lock:
                self._session_results[session_id].append(
                    BackendEnvelope(
                        session_id=session_id,
                        kind="log",
                        payload={"message": "audio chunk dropped due to backpressure"},
                        created_at=time.time(),
                    )
                )
            return False

    def poll_results(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            session_results = self._session_results.get(session_id)
            if not session_results:
                return []
            items: list[dict[str, Any]] = []
            while session_results:
                envelope = session_results.popleft()
                items.append(
                    {
                        "kind": envelope.kind,
                        "payload": envelope.payload,
                        "created_at": envelope.created_at,
                    }
                )
            return items

    def close_session(self, session_id: str) -> None:
        self._put_command(
            BackendCommand(
                kind="close_session",
                session_id=session_id,
                created_at=time.time(),
            )
        )
        with self._lock:
            self._session_results.pop(session_id, None)

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            commands = self._commands
            results = self._results

        if commands is not None:
            try:
                commands.put_nowait(
                    BackendCommand(
                        kind="shutdown",
                        session_id="",
                        created_at=time.time(),
                    )
                )
            except queue.Full:
                pass

        self._stop_dispatcher.set()
        if process is not None:
            process.join(timeout=2)

        dispatcher = self._dispatcher
        if dispatcher is not None:
            dispatcher.join(timeout=1)

        with self._lock:
            self._process = None
            self._commands = None
            self._results = None
            self._dispatcher = None
            self._session_results.clear()

    def _put_command(self, command: BackendCommand) -> None:
        self.start()
        if self._commands is None:
            raise RuntimeError("backend bridge not started")
        self._commands.put(command)

    def _dispatch_results(self) -> None:
        if self._results is None:
            return

        while not self._stop_dispatcher.is_set():
            try:
                envelope: BackendEnvelope = self._results.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                self._session_results[envelope.session_id].append(envelope)


def decode_pcm_s16le_to_f32(pcm_s16le: bytes) -> array:
    pcm = array("h")
    pcm.frombytes(pcm_s16le[: len(pcm_s16le) - (len(pcm_s16le) % 2)])
    if pcm.itemsize != 2:
        raise RuntimeError("unexpected int16 itemsize")
    if sys.byteorder != "little":
        pcm.byteswap()
    return array("f", (sample / 32768.0 for sample in pcm))


def format_inference_log(result: InferenceResult) -> str:
    if result.error:
        return f"{result.model_name} failed at {result.trigger_seconds:.1f}s: {result.error}"
    return (
        f"{result.model_name} completed at {result.trigger_seconds:.1f}s "
        f"with payload={result.payload}"
    )


def render_score_plot(history: list[tuple[float, float]]) -> str:
    width = 720
    height = 240
    pad = 28
    inner_width = width - (pad * 2)
    inner_height = height - (pad * 2)

    if not history:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="{width}" height="{height}" fill="#fffaf0"/>'
            f'<text x="{pad}" y="{height / 2}" fill="#8a4b08" font-size="14">No data yet</text>'
            "</svg>"
        )
        return encode_svg(svg)

    xs = [point[0] for point in history]
    ys = [point[1] for point in history]
    min_y = min(ys)
    max_y = max(ys)
    span_y = max(max_y - min_y, 1e-6)
    span_x = max(xs[-1] - xs[0], 1e-6)

    points = []
    for index, (x_val, y_val) in enumerate(history):
        x = pad if len(history) == 1 else pad + ((x_val - xs[0]) / span_x) * inner_width
        y = pad + inner_height - ((y_val - min_y) / span_y) * inner_height
        if len(history) == 1:
            x = pad + (index * 0)
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    latest = history[-1][1]
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" rx="16" fill="#fffaf0"/>
  <rect x="{pad}" y="{pad}" width="{inner_width}" height="{inner_height}" fill="#fff" stroke="#d8c1a0"/>
  <polyline fill="none" stroke="#c2410c" stroke-width="3" points="{polyline}" />
  <text x="{pad}" y="18" fill="#7c2d12" font-size="14" font-family="monospace">short_window_model score</text>
  <text x="{pad}" y="{height - 8}" fill="#7c2d12" font-size="12" font-family="monospace">latest={latest:.4f}</text>
</svg>
""".strip()
    return encode_svg(svg)


def encode_svg(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


bridge = BackendBridge()
atexit.register(bridge.shutdown)
