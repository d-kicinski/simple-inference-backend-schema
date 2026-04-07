from __future__ import annotations

import unittest

from backend import ModelSchedule, StreamingInferenceBackend


class StreamingInferenceBackendTests(unittest.TestCase):
    def test_schedules_fire_at_expected_intervals(self) -> None:
        observed: dict[str, list[tuple[int, int]]] = {
            "short": [],
            "medium": [],
            "long": [],
        }

        def capture(name: str):
            def _runner(samples, sample_rate: int):
                observed[name].append((len(samples), sample_rate))
                return {"samples": len(samples)}

            return _runner

        backend = StreamingInferenceBackend(
            sample_rate=4,
            max_buffer_seconds=30,
            executor_workers=1,
            schedules=[
                ModelSchedule("short", cadence_seconds=1, window_seconds=2, runner=capture("short")),
                ModelSchedule("medium", cadence_seconds=10, window_seconds=10, runner=capture("medium")),
                ModelSchedule("long", cadence_seconds=30, window_seconds=30, runner=capture("long")),
            ],
        )

        for _ in range(31):
            backend.push_audio_frame([0.1, 0.2, 0.3, 0.4])

        backend.close()

        self.assertEqual(len(observed["short"]), 30)
        self.assertEqual(len(observed["medium"]), 3)
        self.assertEqual(len(observed["long"]), 1)
        self.assertTrue(all(size == 8 for size, _ in observed["short"]))
        self.assertTrue(all(size == 40 for size, _ in observed["medium"]))
        self.assertTrue(all(size == 120 for size, _ in observed["long"]))

    def test_drain_results_exposes_model_outputs(self) -> None:
        backend = StreamingInferenceBackend(
            sample_rate=2,
            max_buffer_seconds=30,
            executor_workers=1,
            schedules=[
                ModelSchedule(
                    "short",
                    cadence_seconds=1,
                    window_seconds=2,
                    runner=lambda samples, _: {"sum": round(sum(samples), 2)},
                )
            ],
        )

        backend.push_audio_frame([0.5, 0.5])
        backend.push_audio_frame([1.0, 1.0])
        backend.close()

        results = backend.drain_results()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].model_name, "short")
        self.assertEqual(results[0].window_start_seconds, 0.0)
        self.assertEqual(results[0].window_end_seconds, 2.0)
        self.assertEqual(results[0].payload, {"sum": 3.0})


if __name__ == "__main__":
    unittest.main()
