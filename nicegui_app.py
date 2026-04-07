from __future__ import annotations

from demo_runtime import DemoController, format_result

try:
    from nicegui import ui
except ImportError as exc:  # pragma: no cover - import guard for missing optional dependency
    raise SystemExit(
        "NiceGUI is not installed. Run `python3 -m pip install nicegui` first."
    ) from exc


def main() -> None:
    controller = DemoController()
    latest_by_model: dict[str, str] = {}

    def refresh_latest() -> None:
        if not latest_by_model:
            latest_summary.set_text("No results yet.")
            return
        latest_summary.set_text(
            "\n".join(
                f"{name}: {payload}"
                for name, payload in sorted(latest_by_model.items())
            )
        )

    def start_stream() -> None:
        controller.start()
        status_label.set_text(controller.status_text(latest_by_model))

    def stop_stream() -> None:
        controller.stop()
        status_label.set_text(controller.status_text(latest_by_model))

    def poll_results() -> None:
        for result in controller.drain_results():
            latest_by_model[result.model_name] = str(
                result.payload if not result.error else result.error
            )
            log.push(format_result(result))
        refresh_latest()
        status_label.set_text(controller.status_text(latest_by_model))

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4"):
        ui.label("NiceGUI streaming backend demo").classes("text-2xl font-bold")
        ui.label(
            "The backend runs in worker threads and only pushes results to a queue. "
            "This UI polls that queue on the UI side."
        ).classes("text-sm text-slate-700")
        with ui.row().classes("gap-2"):
            ui.button("Start stream", on_click=start_stream)
            ui.button("Stop stream", on_click=stop_stream)
        status_label = ui.label("stream=idle, latest_models=0").classes("font-mono")
        latest_summary = ui.label("No results yet.").classes("font-mono whitespace-pre")
        log = ui.log(max_lines=200).classes("w-full h-96")

    ui.timer(0.2, poll_results)
    start_stream()
    ui.run(title="NiceGUI Backend Demo")


if __name__ in {"__main__", "__mp_main__"}:
    main()
