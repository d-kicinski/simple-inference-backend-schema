from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import torch

from backend import ModelSchedule, StreamingInferenceBackend


class TinyAudioModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def run_threadpool_probe(job_count: int = 8, max_workers: int = 4) -> dict[str, object]:
    model = TinyAudioModel().eval()

    def infer(seed: int) -> list[float]:
        generator = torch.Generator().manual_seed(seed)
        audio = torch.randn((1, 1, 32_000), generator=generator)
        with torch.inference_mode():
            output = model(audio)
        return output.squeeze(0).tolist()

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        outputs = list(executor.map(infer, range(job_count)))
    elapsed = time.perf_counter() - started_at

    return {
        "job_count": job_count,
        "max_workers": max_workers,
        "elapsed_seconds": round(elapsed, 4),
        "sample_output": [round(value, 6) for value in outputs[0]],
    }


def run_backend_probe() -> dict[str, object]:
    torch.set_num_threads(1)
    model = TinyAudioModel().eval()

    def runner(samples, sample_rate: int) -> dict[str, object]:
        tensor = torch.tensor(samples, dtype=torch.float32).reshape(1, 1, -1)
        with torch.inference_mode():
            output = model(tensor).squeeze(0)
        return {
            "sample_rate": sample_rate,
            "scores": [round(value, 6) for value in output.tolist()],
        }

    backend = StreamingInferenceBackend(
        sample_rate=16_000,
        max_buffer_seconds=30,
        schedules=[
            ModelSchedule(
                name="torch-short",
                cadence_seconds=1,
                window_seconds=2,
                runner=runner,
            )
        ],
        executor_workers=2,
    )

    for _ in range(3):
        backend.push_audio_frame([0.1] * 16_000)

    backend.close()
    results = backend.drain_results()

    return {
        "result_count": len(results),
        "payload": results[0].payload if results else None,
        "errors": [result.error for result in results if result.error],
    }


def main() -> None:
    print("direct_threadpool_probe", run_threadpool_probe())
    print("backend_threadpool_probe", run_backend_probe())
    print("torch_cuda_available", torch.cuda.is_available())


if __name__ == "__main__":
    main()
