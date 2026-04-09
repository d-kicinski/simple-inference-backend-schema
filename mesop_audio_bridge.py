from __future__ import annotations

from typing import Any, Callable

import mesop as me


@me.web_component(path="./mesop_audio_bridge.js")
def audio_bridge(
    *,
    session_id: str,
    ingest_url: str,
    results_url: str,
    recording: bool,
    on_results: Callable[[me.WebEvent], Any] | None = None,
    on_status: Callable[[me.WebEvent], Any] | None = None,
    key: str | None = None,
):
    return me.insert_web_component(
        name="audio-recorder-bridge",
        key=key,
        events={
            "resultsEvent": on_results,
            "statusEvent": on_status,
        },
        properties={
            "sessionId": session_id,
            "ingestUrl": ingest_url,
            "resultsUrl": results_url,
            "recording": recording,
        },
    )
