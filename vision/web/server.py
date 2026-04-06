from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vision.state import SharedFrameState

LOGGER = logging.getLogger("astra_vision")


def build_handler(state: SharedFrameState):
    class VisionRequestHandler(BaseHTTPRequestHandler):
        server_version = "AstraVisionHTTP/1.0"

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            LOGGER.info("%s - %s", self.address_string(), format % args)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send_index()
                return
            if self.path == "/api/status":
                self._send_json(state.snapshot())
                return
            if self.path == "/stream.mjpg":
                self._send_stream()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def _send_index(self) -> None:
            html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Astra YOLO Vision</title>
  <style>
    :root {
      --bg: #09111f;
      --panel: rgba(18, 24, 38, 0.84);
      --panel-border: rgba(255, 255, 255, 0.09);
      --text: #ecf2ff;
      --muted: #a9b6d3;
      --accent: #34d399;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(96, 165, 250, 0.22), transparent 30%),
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.16), transparent 26%),
        linear-gradient(160deg, #040816 0%, #0b1220 44%, #111b2f 100%);
    }
    .shell {
      max-width: 1360px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: minmax(0, 1.75fr) minmax(320px, 0.75fr);
      gap: 20px;
    }
    .hero, .panel {
      border: 1px solid var(--panel-border);
      background: var(--panel);
      border-radius: 24px;
      backdrop-filter: blur(12px);
      overflow: hidden;
    }
    .hero { padding: 18px; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 4vw, 46px); }
    p { margin: 0; color: var(--muted); line-height: 1.6; }
    .stream-wrap { margin-top: 18px; border-radius: 16px; overflow: hidden; }
    .stream-wrap img { width: 100%; display: block; aspect-ratio: 4/3; background: #030611; }
    .panel { padding: 18px; }
    .row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed rgba(255,255,255,0.13); }
    .row:last-child { border-bottom: 0; }
    .k { color: var(--muted); }
    .v { font-weight: 700; }
    @media (max-width: 1040px) { .shell { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Astra + YOLO</h1>
      <p>使用你提供的 .bin 模型做实时检测，叠加结果并通过网页展示。</p>
      <div class="stream-wrap"><img src="/stream.mjpg" alt="stream"/></div>
    </section>
    <aside class="panel">
      <div class="row"><span class="k">状态</span><span class="v" id="s">starting</span></div>
      <div class="row"><span class="k">FPS</span><span class="v" id="fps">0.0</span></div>
      <div class="row"><span class="k">Frame</span><span class="v" id="f">0</span></div>
      <div class="row"><span class="k">Detections</span><span class="v" id="d">0</span></div>
    </aside>
  </main>
  <script>
    async function tick() {
      try {
        const r = await fetch('/api/status', {cache: 'no-store'});
        const s = await r.json();
        document.getElementById('s').textContent = s.status;
        document.getElementById('fps').textContent = Number(s.fps || 0).toFixed(1);
        document.getElementById('f').textContent = String(s.frame_index || 0);
        document.getElementById('d').textContent = String((s.detections || []).length);
      } catch (_) {
        document.getElementById('s').textContent = 'offline';
      }
    }
    tick();
    setInterval(tick, 1000);
  </script>
</body>
</html>"""
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            frame_index = -1
            while True:
                frame_index, jpeg = state.wait_for_frame(frame_index, timeout=1.0)
                if jpeg is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

    return VisionRequestHandler


class VisionWebServer:
    def __init__(self, state: SharedFrameState, host: str, port: int) -> None:
        self._server = ThreadingHTTPServer((host, port), build_handler(state))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="vision-web")
        self.host = host
        self.port = port

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
