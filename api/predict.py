"""
Vercel Python serverless function — POST /api/predict
Receives a 30-frame keypoint sequence, returns the predicted sign.
"""

import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np
import onnxruntime as ort

# ── Paths (relative to repo root) ────────────────────────────
_ROOT    = os.path.join(os.path.dirname(__file__), "..")
_MODEL   = os.path.join(_ROOT, "lstm_sign_model.onnx")
_ACTIONS = os.path.join(_ROOT, "actions.npy")

# ── Load once per container (warm starts reuse this) ─────────
_session = None
_actions = None
_input_name = None

def _load():
    global _session, _actions, _input_name
    if _session is None:
        _session    = ort.InferenceSession(_MODEL, providers=["CPUExecutionProvider"])
        _actions    = np.load(_ACTIONS, allow_pickle=True)
        _input_name = _session.get_inputs()[0].name

def _cors(handler):
    handler.send_header("Access-Control-Allow-Origin",  "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")

# ── Vercel handler ─────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass  # silence access logs

    def do_OPTIONS(self):
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_POST(self):
        try:
            _load()

            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))

            # sequence: list[list[float]]  shape (30, 126)
            seq = np.array(body["sequence"], dtype=np.float32)
            if seq.shape != (30, 126):
                raise ValueError(f"Expected shape (30,126), got {seq.shape}")

            inp   = seq[np.newaxis, ...]                          # (1, 30, 126)
            probs = _session.run(None, {_input_name: inp})[0][0] # (n_classes,)

            mx   = int(np.argmax(probs))
            conf = float(probs[mx])
            sign = str(_actions[mx]) if conf > 0.50 else ""

            result = json.dumps({
                "sign":       sign,
                "confidence": round(conf, 4),
                "label_idx":  mx,
            }).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            _cors(self)
            self.end_headers()
            self.wfile.write(result)

        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            _cors(self)
            self.end_headers()
            self.wfile.write(err)
