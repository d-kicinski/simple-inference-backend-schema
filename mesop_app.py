from dataclasses import field
import uuid

from absl.flags import FLAGS
from flask import Flask, jsonify, request
import mesop as me
from mesop.server.wsgi_app import create_app

from mesop_audio_bridge import audio_bridge
from mesop_backend_bridge import EXPECTED_SAMPLE_RATE, bridge


API_AUDIO_PATH = "/api/audio"
API_RESULTS_PATH = "/api/results"


@me.stateclass
class AppState:
    session_id: str = ""
    recording: bool = False
    status: str = "idle"
    latest_text: str = "No long-window text yet."
    latest_plot_url: str = ""
    logs: list[str] = field(default_factory=list)


def on_load(_: me.LoadEvent) -> None:
    state = me.state(AppState)
    if state.session_id:
        return
    state.session_id = uuid.uuid4().hex
    state.status = "session-ready"
    bridge.open_session(state.session_id)


def on_results(event: me.WebEvent) -> None:
    payload = event.value or {}
    state = me.state(AppState)

    for item in payload.get("items", []):
        kind = item.get("kind")
        item_payload = item.get("payload", {})
        if kind == "plot":
            state.latest_plot_url = item_payload.get("image_url", "")
        elif kind == "text":
            state.latest_text = item_payload.get("summary", state.latest_text)
        elif kind in {"log", "status"}:
            message = item_payload.get("message")
            if message:
                state.logs.append(message)

    if len(state.logs) > 16:
        state.logs = state.logs[-16:]


def on_status(event: me.WebEvent) -> None:
    payload = event.value or {}
    state = me.state(AppState)
    status = payload.get("status")
    if status:
        state.status = status
    error = payload.get("error")
    if error:
        state.logs.append(f"browser error: {error}")
        state.logs = state.logs[-16:]


def start_recording(_: me.ClickEvent) -> None:
    state = me.state(AppState)
    state.recording = True
    state.status = "starting"


def stop_recording(_: me.ClickEvent) -> None:
    state = me.state(AppState)
    state.recording = False
    state.status = "stopping"


@me.page(path="/", title="Mesop Audio Streaming PoC", on_load=on_load)
def page() -> None:
    state = me.state(AppState)

    with me.box(
        style=me.Style(
            background="#f7f1e8",
            min_height="100vh",
            padding=me.Padding(top=24, right=24, bottom=24, left=24),
        )
    ):
        with me.box(
            style=me.Style(
                display="flex",
                flex_direction="column",
                gap=20,
                max_width="960px",
                margin=me.Margin(left="auto", right="auto"),
            )
        ):
            me.text("Mesop Audio Streaming PoC", type="headline-4")
            me.markdown(
                """
This page keeps Mesop off the hot audio path:

- browser audio chunks go to `/api/audio`
- a separate backend process runs inference and artifact workers
- browser polling reads `/api/results`
- Mesop only updates state when result batches arrive
                """.strip()
            )

            with me.box(style=me.Style(display="flex", gap=12)):
                me.button("Start recording", on_click=start_recording, type="raised")
                me.button("Stop recording", on_click=stop_recording, type="stroked")

            me.text(f"session={state.session_id or 'pending'}", style=me.Style(font_family="monospace"))
            me.text(f"status={state.status}", style=me.Style(font_family="monospace"))

            audio_bridge(
                key="audio-bridge",
                session_id=state.session_id,
                ingest_url=API_AUDIO_PATH,
                results_url=API_RESULTS_PATH,
                recording=state.recording,
                on_results=on_results,
                on_status=on_status,
            )

            with me.box(
                style=me.Style(
                    display="grid",
                    grid_template_columns="1fr 1fr",
                    gap=16,
                )
            ):
                with me.box(
                    style=me.Style(
                        background="#fffdf7",
                        border_radius=16,
                        padding=me.Padding(top=16, right=16, bottom=16, left=16),
                    )
                ):
                    me.text("Long-window text", type="headline-6")
                    me.markdown(state.latest_text)
                with me.box(
                    style=me.Style(
                        background="#fffdf7",
                        border_radius=16,
                        padding=me.Padding(top=16, right=16, bottom=16, left=16),
                    )
                ):
                    me.text("Short-window plot", type="headline-6")
                    if state.latest_plot_url:
                        me.image(
                            src=state.latest_plot_url,
                            alt="short-window plot",
                            style=me.Style(width="100%"),
                        )
                    else:
                        me.text("No plot yet.", style=me.Style(font_family="monospace"))

            with me.box(
                style=me.Style(
                    background="#fffdf7",
                    border_radius=16,
                    padding=me.Padding(top=16, right=16, bottom=16, left=16),
                )
            ):
                me.text("Recent logs", type="headline-6")
                for line in state.logs[-12:]:
                    me.text(line, style=me.Style(font_family="monospace", font_size=13))


def configure_api(flask_app: Flask) -> None:
    @flask_app.post(API_AUDIO_PATH)
    def ingest_audio():
        session_id = request.headers.get("X-Session-Id", "")
        if not session_id:
            return jsonify({"error": "missing X-Session-Id"}), 400

        payload = request.get_data(cache=False)
        sample_rate = int(request.headers.get("X-Sample-Rate", EXPECTED_SAMPLE_RATE))
        channels = int(request.headers.get("X-Channels", 1))
        accepted = bridge.push_audio_chunk(
            session_id=session_id,
            pcm_s16le=payload,
            sample_rate=sample_rate,
            channels=channels,
        )
        return jsonify({"accepted": accepted}), 202 if accepted else 429

    @flask_app.get(API_RESULTS_PATH)
    def fetch_results():
        session_id = request.headers.get("X-Session-Id", "")
        if not session_id:
            return jsonify({"error": "missing X-Session-Id"}), 400
        return jsonify({"items": bridge.poll_results(session_id)})


app = create_app(prod_mode=False)
configure_api(app._flask_app)


if __name__ == "__main__":
    if not FLAGS.is_parsed():
        FLAGS(["mesop_app.py"])
    bridge.start()
    app.run()
