"""
Local dev server for testing the Vercel app before deploying.
Serves index.html + actions.json as static files, and runs
api/predict.py logic at /api/predict — all on one port.

    python dev_server.py
    # Open http://localhost:3000
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# Add api/ to path so we can import predict logic directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

import numpy as np
import onnxruntime as ort

# ── Load model once ───────────────────────────────────────────
_ROOT    = os.path.dirname(__file__)
_MODEL   = os.path.join(_ROOT, "lstm_sign_model.onnx")
_ACTIONS = os.path.join(_ROOT, "actions.npy")
_session     = ort.InferenceSession(_MODEL, providers=["CPUExecutionProvider"])
_actions     = np.load(_ACTIONS, allow_pickle=True)
_input_name  = _session.get_inputs()[0].name
print(f"✅ Model loaded — {len(_actions)} signs")

# ── Static file map ───────────────────────────────────────────
STATIC = {
    "/":             ("index.html",  "text/html"),
    "/index.html":   ("index.html",  "text/html"),
    "/actions.json": ("actions.json","application/json"),
}

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.path}  {args[0] if args else ''}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        entry = STATIC.get(self.path)
        if entry:
            fpath, mime = entry
            fpath = os.path.join(_ROOT, fpath)
            if os.path.exists(fpath):
                data = open(fpath, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self._cors()
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not found")

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_response(404); self.end_headers(); return

        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))

        seq  = np.array(body["sequence"], dtype=np.float32)
        inp  = seq[np.newaxis, ...]
        probs = _session.run(None, {_input_name: inp})[0][0]
        mx   = int(np.argmax(probs))
        conf = float(probs[mx])
        sign = str(_actions[mx]) if conf > 0.50 else ""

        result = json.dumps({"sign": sign, "confidence": round(conf, 4), "label_idx": mx}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(result)

PORT = 3000
print(f"🚀 Dev server → http://localhost:{PORT}")
HTTPServer(("", PORT), _Handler).serve_forever()
