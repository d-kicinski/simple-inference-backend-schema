from __future__ import annotations

import asyncio

from demo_runtime import DemoController, format_result

try:
    import flet as ft
except ImportError as exc:  # pragma: no cover - import guard for missing optional dependency
    raise SystemExit(
        "Flet is not installed. Run `python3 -m pip install flet` first."
    ) from exc


def main(page: ft.Page) -> None:
    controller = DemoController()
    latest_by_model: dict[str, str] = {}

    page.title = "Flet Backend Demo"
    page.padding = 24
    page.window_width = 900
    page.window_height = 700

    status_text = ft.Text("stream=idle, latest_models=0", font_family="monospace")
    latest_text = ft.Text("No results yet.", selectable=True, font_family="monospace")
    log_view = ft.ListView(expand=True, auto_scroll=True, spacing=4)

    def refresh_status() -> None:
        status_text.value = controller.status_text(latest_by_model)

    def refresh_latest() -> None:
        if not latest_by_model:
            latest_text.value = "No results yet."
        else:
            latest_text.value = "\n".join(
                f"{name}: {payload}"
                for name, payload in sorted(latest_by_model.items())
            )

    def append_result_line(message: str) -> None:
        log_view.controls.append(ft.Text(message, selectable=True))
        if len(log_view.controls) > 200:
            log_view.controls = log_view.controls[-200:]

    def start_stream(_: ft.ControlEvent | None = None) -> None:
        controller.start()
        refresh_status()
        page.update()

    def stop_stream(_: ft.ControlEvent | None = None) -> None:
        controller.stop()
        refresh_status()
        page.update()

    async def poll_results() -> None:
        while True:
            changed = False
            for result in controller.drain_results():
                latest_by_model[result.model_name] = str(
                    result.payload if not result.error else result.error
                )
                append_result_line(format_result(result))
                changed = True

            refresh_latest()
            refresh_status()
            start_button.disabled = controller.is_running()
            stop_button.disabled = not controller.is_running()

            if changed or controller.is_running() or controller.finished.is_set():
                page.update()

            await asyncio.sleep(0.2)

    start_button = ft.Button("Start stream", on_click=start_stream)
    stop_button = ft.Button("Stop stream", on_click=stop_stream, disabled=True)

    page.add(
        ft.Text("Flet streaming backend demo", size=24, weight=ft.FontWeight.BOLD),
        ft.Text(
            "The backend runs in worker threads and only pushes results to a queue. "
            "The page task polls that queue and updates controls on the UI side."
        ),
        ft.Row([start_button, stop_button]),
        status_text,
        ft.Container(latest_text, padding=12, border=ft.Border.all(1, ft.Colors.OUTLINE)),
        ft.Container(log_view, expand=True, padding=12, border=ft.Border.all(1, ft.Colors.OUTLINE)),
    )

    page.run_task(poll_results)
    start_stream()


if __name__ == "__main__":
    ft.run(main)
