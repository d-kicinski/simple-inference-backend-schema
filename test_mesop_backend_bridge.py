from __future__ import annotations

from array import array
import time
import unittest

from mesop_backend_bridge import BackendBridge, decode_pcm_s16le_to_f32


class MesopBackendBridgeTests(unittest.TestCase):
    def test_decode_pcm_s16le_to_f32(self) -> None:
        pcm = array("h", [-32768, 0, 32767]).tobytes()

        decoded = decode_pcm_s16le_to_f32(pcm)

        self.assertEqual(len(decoded), 3)
        self.assertAlmostEqual(decoded[0], -1.0, places=4)
        self.assertAlmostEqual(decoded[1], 0.0, places=4)
        self.assertAlmostEqual(decoded[2], 32767 / 32768.0, places=4)

    def test_backend_bridge_produces_plot_and_text_updates(self) -> None:
        bridge = BackendBridge(queue_size=512)
        session_id = "test-session"
        chunk = array("h", [2000] * 16_000).tobytes()

        try:
            bridge.open_session(session_id)
            for _ in range(31):
                accepted = bridge.push_audio_chunk(
                    session_id=session_id,
                    pcm_s16le=chunk,
                    sample_rate=16_000,
                    channels=1,
                )
                self.assertTrue(accepted)

            items = self._wait_for_results(bridge, session_id)
        finally:
            bridge.shutdown()

        kinds = {item["kind"] for item in items}
        self.assertIn("plot", kinds)
        self.assertIn("text", kinds)

        plot_item = next(item for item in items if item["kind"] == "plot")
        text_item = next(item for item in items if item["kind"] == "text")

        self.assertTrue(plot_item["payload"]["image_url"].startswith("data:image/svg+xml;base64,"))
        self.assertIn("30s summary", text_item["payload"]["summary"])

    def _wait_for_results(
        self,
        bridge: BackendBridge,
        session_id: str,
        timeout_seconds: float = 5.0,
    ) -> list[dict]:
        deadline = time.time() + timeout_seconds
        items: list[dict] = []

        while time.time() < deadline:
            items.extend(bridge.poll_results(session_id))
            kinds = {item["kind"] for item in items}
            if "plot" in kinds and "text" in kinds:
                return items
            time.sleep(0.1)

        self.fail(f"timed out waiting for plot/text results, got: {items}")


if __name__ == "__main__":
    unittest.main()
