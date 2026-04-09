class AudioRecorderBridge extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.sessionId = "";
    this.ingestUrl = "";
    this.resultsUrl = "";
    this.recording = false;
    this.mediaStream = null;
    this.audioContext = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.pendingSamples = [];
    this.flushTimer = null;
    this.pollTimer = null;
    this.sendInFlight = false;
    this.status = "idle";
    this.lastError = "";
    this.render();
  }

  connectedCallback() {
    this.render();
    this.startPolling();
    if (this.recording) {
      this.startRecording();
    }
  }

  disconnectedCallback() {
    this.stopRecording();
    window.clearInterval(this.pollTimer);
    window.clearInterval(this.flushTimer);
  }

  set recording(value) {
    this._recording = Boolean(value);
    if (this.isConnected) {
      if (this._recording) {
        this.startRecording();
      } else {
        this.stopRecording();
      }
    }
  }

  get recording() {
    return this._recording;
  }

  async startRecording() {
    if (this.mediaStream || !this.sessionId || !this.ingestUrl) {
      return;
    }

    try {
      this.setStatus("initializing");
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        throw new Error("AudioContext is not available in this browser");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContextClass({ sampleRate: 16000 });
      await audioContext.resume();
      const sourceNode = audioContext.createMediaStreamSource(stream);
      const processorNode = audioContext.createScriptProcessor(4096, 1, 1);

      processorNode.onaudioprocess = (event) => {
        const channelData = event.inputBuffer.getChannelData(0);
        this.pendingSamples.push(new Float32Array(channelData));
      };

      sourceNode.connect(processorNode);
      processorNode.connect(audioContext.destination);

      this.mediaStream = stream;
      this.audioContext = audioContext;
      this.sourceNode = sourceNode;
      this.processorNode = processorNode;
      this.flushTimer = window.setInterval(() => this.flushAudio(), 150);
      this.setStatus("recording");
    } catch (error) {
      this.lastError = String(error);
      this.setStatus("error");
    }
  }

  stopRecording() {
    if (this.flushTimer) {
      window.clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode.onaudioprocess = null;
      this.processorNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop();
      }
      this.mediaStream = null;
    }
    this.pendingSamples = [];
    if (this.status !== "idle") {
      this.setStatus("idle");
    }
  }

  async flushAudio() {
    if (this.sendInFlight || this.pendingSamples.length === 0) {
      return;
    }

    const chunk = this.mergePendingSamples();
    if (!chunk) {
      return;
    }

    this.sendInFlight = true;
    try {
      const pcm = this.floatTo16BitPCM(chunk);
      await fetch(this.ingestUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Session-Id": this.sessionId,
          "X-Sample-Rate": "16000",
          "X-Channels": "1",
        },
        body: pcm,
      });
    } catch (error) {
      this.lastError = String(error);
      this.setStatus("error");
    } finally {
      this.sendInFlight = false;
    }
  }

  mergePendingSamples() {
    if (this.pendingSamples.length === 0) {
      return null;
    }
    let totalLength = 0;
    for (const chunk of this.pendingSamples) {
      totalLength += chunk.length;
    }
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.pendingSamples) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.pendingSamples = [];
    return merged;
  }

  floatTo16BitPCM(floatBuffer) {
    const pcm = new Int16Array(floatBuffer.length);
    for (let index = 0; index < floatBuffer.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, floatBuffer[index]));
      pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return pcm.buffer;
  }

  startPolling() {
    if (this.pollTimer) {
      return;
    }
    this.pollTimer = window.setInterval(() => this.pollResults(), 500);
  }

  async pollResults() {
    if (!this.resultsUrl || !this.sessionId) {
      return;
    }
    try {
      const response = await fetch(this.resultsUrl, {
        method: "GET",
        headers: { "X-Session-Id": this.sessionId },
      });
      const payload = await response.json();
      if (payload.items && payload.items.length > 0) {
        this.dispatchEvent(
          new CustomEvent("resultsEvent", {
            detail: payload,
            bubbles: true,
            composed: true,
          }),
        );
      }
    } catch (error) {
      this.lastError = String(error);
      this.setStatus("error");
    }
  }

  setStatus(nextStatus) {
    this.status = nextStatus;
    this.render();
    this.dispatchEvent(
      new CustomEvent("statusEvent", {
        detail: {
          sessionId: this.sessionId,
          status: this.status,
          error: this.lastError,
        },
        bubbles: true,
        composed: true,
      }),
    );
  }

  render() {
    const label = this.status === "recording" ? "Recording..." : "Recorder ready";
    const error = this.lastError ? `<div class="error">${this.lastError}</div>` : "";
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          border: 1px solid #d6c6ad;
          border-radius: 16px;
          background: #fffdf7;
          padding: 16px;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .title {
          color: #7c2d12;
          font-size: 14px;
          margin-bottom: 8px;
        }
        .status {
          color: #92400e;
          font-size: 13px;
        }
        .error {
          color: #b91c1c;
          margin-top: 8px;
          white-space: pre-wrap;
        }
      </style>
      <div class="title">Browser audio bridge</div>
      <div class="status">${label}</div>
      ${error}
    `;
  }
}

customElements.define("audio-recorder-bridge", AudioRecorderBridge);
