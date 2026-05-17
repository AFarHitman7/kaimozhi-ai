"""
KaiMozhi — Zero-Rerender MJPEG Frontend
-----------------------------------------
Architecture:
  • OpenCV + MediaPipe + LSTM run in a background daemon thread
  • Flask serves MJPEG on /video_feed and JSON state on /state
  • One components.html block renders video + live info via JS fetch()
  • Streamlit NEVER auto-rerenders → perfectly smooth stream

Run:  streamlit run app.py
"""

import json
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from flask import Flask, Response, jsonify, request
from tensorflow.keras.models import load_model

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="KaiMozhi · Live Sign Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal Streamlit CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.stApp{background:#0d0f14;color:#e8eaf0}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#131720,#0d0f14);border-right:1px solid #1e2230}
.brand-title{font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:700;
  background:linear-gradient(135deg,#7c8fff,#b87cff,#ff7cb0);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0}
.brand-sub{font-size:.8rem;color:#5a6177;letter-spacing:.08em;text-transform:uppercase;margin:4px 0 28px}
.section-heading{font-family:'Space Grotesk',sans-serif;font-size:.9rem;font-weight:600;
  color:#7c8fff;text-transform:uppercase;letter-spacing:.12em;
  margin:20px 0 10px;border-bottom:1px solid #252b40;padding-bottom:6px}
.pill{display:inline-block;padding:3px 12px;border-radius:50px;font-size:.76rem;font-weight:500;margin:2px}
.pill-green{background:#0d2b1e;color:#4dffb4;border:1px solid #4dffb466}
.pill-yellow{background:#2b2300;color:#ffd84d;border:1px solid #ffd84d66}
.pill-blue{background:#0d1a2b;color:#4da6ff;border:1px solid #4da6ff66}
.stButton>button{background:linear-gradient(135deg,#7c8fff,#b87cff);color:#fff;border:none;
  border-radius:9px;padding:9px 24px;font-weight:600;transition:opacity .2s,transform .15s}
.stButton>button:hover{opacity:.88;transform:translateY(-1px)}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────
BASE         = Path(__file__).parent
MODEL_PATH   = BASE / "lstm_sign_model.keras"
ACTIONS_PATH = BASE / "actions.npy"
MJPEG_PORT   = 7860
_ENC_PARAMS  = [cv2.IMWRITE_JPEG_QUALITY, 70]

# ── Shared state ──────────────────────────────────────────────
@dataclass
class _AppState:
    # Single Condition: guards jpeg/sign/conf/history AND wakes MJPEG generators
    frame_cond:     threading.Condition = field(default_factory=threading.Condition)
    current_jpeg:   Optional[bytes]     = None
    current_sign:   str                 = ""
    current_conf:   float               = 0.0
    history:        List[dict]          = field(default_factory=list)
    total_detected: int                 = 0
    cam_error:      str                 = ""
    # Settings — separate lock so MJPEG generators never contend here
    settings_lock:  threading.Lock      = field(default_factory=threading.Lock)
    conf_threshold: float               = 0.70
    stability:      int                 = 4
    predict_every:  int                 = 2

_state = _AppState()

# ── Keypoint helpers ──────────────────────────────────────────
def _normalize(kp: np.ndarray) -> np.ndarray:
    out = kp.copy().reshape(2, 21, 3)
    for h in range(2):
        hand = out[h]
        if not np.allclose(hand, 0):
            hand -= hand[0].copy()
            scale = np.linalg.norm(hand[9] - hand[0]) + 1e-6
            hand /= scale
            out[h] = hand
    return out.flatten()

def _extract(results) -> np.ndarray:
    lh, rh = np.zeros(63, np.float32), np.zeros(63, np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hl in enumerate(results.multi_hand_landmarks):
            lab    = results.multi_handedness[i].classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark]).flatten().astype(np.float32)
            if lab == "Left": lh = coords
            else:             rh = coords
    return _normalize(np.concatenate([lh, rh]))

# ── Detection thread ──────────────────────────────────────────
def _detection_loop():
    if not MODEL_PATH.exists():
        _state.cam_error = "Model file not found."
        return

    model   = load_model(str(MODEL_PATH))
    actions = np.load(str(ACTIONS_PATH), allow_pickle=True)
    model(np.zeros((1, 30, int(model.input_shape[-1])), np.float32), training=False)

    @tf.function(reduce_retracing=True)
    def fast_pred(x):
        return model(x, training=False)

    mp_h = mp.solutions.hands
    mp_d = mp.solutions.drawing_utils
    mp_s = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always grab the latest frame

    if not cap.isOpened():
        _state.cam_error = "Cannot open webcam (index 0)."
        return

    sequence      = deque(maxlen=30)
    recent_preds  = deque(maxlen=4)
    current_sign  = ""
    current_conf  = 0.0
    frame_count   = 0
    last_stability = 4

    with mp_h.Hands(static_image_mode=False, min_detection_confidence=0.6,
                    min_tracking_confidence=0.6, max_num_hands=2, model_complexity=0) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)

            with _state.settings_lock:
                conf_threshold = _state.conf_threshold
                stability      = _state.stability
                predict_every  = _state.predict_every

            if stability != last_stability:
                recent_preds   = deque(maxlen=stability)
                last_stability = stability

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_d.draw_landmarks(frame, hl, mp_h.HAND_CONNECTIONS,
                                        mp_s.get_default_hand_landmarks_style(),
                                        mp_s.get_default_hand_connections_style())

            kp = _extract(results)
            sequence.append(kp)
            seq_len = len(sequence)

            if seq_len == 30 and frame_count % predict_every == 0:
                inp   = tf.constant(np.expand_dims(np.array(sequence), 0), dtype=tf.float32)
                probs = fast_pred(inp)[0].numpy()
                mx    = int(np.argmax(probs))
                mp_   = float(probs[mx])
                recent_preds.append(mx if mp_ > conf_threshold else -1)

                if len(recent_preds) == stability:
                    mc, cnt = Counter(recent_preds).most_common(1)[0]
                    if mc != -1 and cnt >= stability - 1:
                        new_sign = str(actions[mc])
                        new_conf = float(probs[mc])
                        with _state.frame_cond:
                            if new_sign != current_sign:
                                _state.history        = [{"sign": new_sign, "conf": new_conf, "ts": time.time()}] + _state.history[:29]
                                _state.total_detected += 1
                        current_sign = new_sign
                        current_conf = new_conf
                    elif mc == -1 and cnt >= stability - 1:
                        current_sign = ""
                        current_conf = 0.0

            # HUD
            h, w = frame.shape[:2]
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (w, 95), (13, 15, 20), -1)
            cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
            lbl   = current_sign or "..."
            color = (100, 220, 160) if current_sign else (100, 100, 120)
            cv2.putText(frame, f"Sign: {lbl}", (18, 58), cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2, cv2.LINE_AA)
            if current_sign:
                bw = int((w - 36) * current_conf)
                cv2.rectangle(frame, (18, 72), (w - 18, 84), (40, 40, 50), -1)
                cv2.rectangle(frame, (18, 72), (18 + bw, 84), (80, 200, 120), -1)
                cv2.putText(frame, f"{current_conf*100:.0f}%", (w - 68, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Buffer: {int(seq_len/30*100)}%", (18, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, "KaiMozhi", (w - 105, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 90, 140), 1, cv2.LINE_AA)

            _, jpeg = cv2.imencode('.jpg', frame, _ENC_PARAMS)
            jpeg_bytes = jpeg.tobytes()

            # Publish and wake all MJPEG generators atomically
            with _state.frame_cond:
                _state.current_jpeg = jpeg_bytes
                _state.current_sign = current_sign
                _state.current_conf = current_conf
                _state.frame_cond.notify_all()

    cap.release()

# ── Flask app ─────────────────────────────────────────────────
_flask_app = Flask(__name__)

def _mjpeg_generator():
    while True:
        with _state.frame_cond:
            _state.frame_cond.wait(timeout=1.0)
            jpg = _state.current_jpeg
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

@_flask_app.route("/video_feed")
def _video_feed():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"})

@_flask_app.route("/state")
def _state_endpoint():
    """Lightweight JSON snapshot — polled by JavaScript every 500 ms."""
    with _state.frame_cond:
        sign    = _state.current_sign
        conf    = _state.current_conf
        history = list(_state.history[:10])
        total   = _state.total_detected
    unique = len({e["sign"] for e in history})
    avg_c  = (sum(e["conf"] for e in history) / len(history) * 100) if history else 0.0
    return jsonify(sign=sign, conf=conf, history=history,
                   total=total, unique=unique, avg_conf=avg_c)

@_flask_app.route("/clear", methods=["POST"])
def _clear():
    with _state.frame_cond:
        _state.history        = []
        _state.total_detected = 0
    return "ok"

@_flask_app.route("/settings", methods=["POST"])
def _settings():
    data = request.get_json(force=True)
    with _state.settings_lock:
        _state.conf_threshold = float(data.get("conf_threshold", _state.conf_threshold))
        _state.stability      = int(data.get("stability",      _state.stability))
        _state.predict_every  = int(data.get("predict_every",  _state.predict_every))
    return "ok"

# ── Start backend once ────────────────────────────────────────
@st.cache_resource
def _start_backend():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(
        target=lambda: _flask_app.run(host="0.0.0.0", port=MJPEG_PORT,
                                      threaded=True, use_reloader=False, debug=False),
        daemon=True,
    ).start()
    threading.Thread(target=_detection_loop, daemon=True).start()
    time.sleep(1.0)
    return True

_start_backend()

# ═══════════════════════════════════════════════════════════════
# Streamlit UI  (static — never auto-rerenders)
# ═══════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="brand-title">🤟 KaiMozhi</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Sign Language AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">⚙️ Detection Settings</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#5a6177;font-size:.82rem;margin-bottom:6px'>
      Adjust sliders — settings apply instantly via the live panel below.
    </p>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small style='color:#3a4060'>KaiMozhi · LSTM + MediaPipe + TF</small>", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px">
<h1 style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;margin:0;
  background:linear-gradient(135deg,#7c8fff,#b87cff,#ff7cb0);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
  Live Sign Language Detection
</h1>
<p style="color:#5a6177;margin:6px 0 0;font-size:.95rem">
  Point your webcam at your hands — KaiMozhi recognises signs in real-time.
</p></div>""", unsafe_allow_html=True)

if _state.cam_error:
    st.error(_state.cam_error, icon="🚫")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model file not found.", icon="🚫")
    st.stop()

actions     = np.load(str(ACTIONS_PATH), allow_pickle=True)
action_list = list(actions)
n_classes   = len(action_list)

st.markdown(
    f'<div style="margin-bottom:16px">'
    f'<span class="pill pill-green">✅ Model Loaded</span>'
    f'<span class="pill pill-blue">🏷 {n_classes} Signs</span>'
    f'<span class="pill pill-blue">📐 LSTM · TF</span>'
    f'<span class="pill pill-yellow">📷 MediaPipe</span>'
    f'<span class="pill pill-green">⚡ MJPEG · Flask</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Single full-width iframe: video + live info, all JS-driven ─
known_pills = "".join(
    f'<span style="display:inline-block;padding:3px 12px;border-radius:50px;'
    f'font-size:.76rem;font-weight:500;margin:2px;background:#0d1a2b;'
    f'color:#4da6ff;border:1px solid #4da6ff66">{a}</span>'
    for a in action_list
)

components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d0f14;color:#e8eaf0;font-family:'Inter',sans-serif;padding:0}}
  .layout{{display:flex;gap:20px;align-items:flex-start}}
  .col-video{{flex:3;min-width:0}}
  .col-info{{flex:2;min-width:260px}}
  .video-wrap{{border-radius:14px;overflow:hidden;background:#0d0f14;border:1px solid #252b40}}
  .video-wrap img{{width:100%;display:block}}
  .sh{{font-size:.82rem;font-weight:600;color:#7c8fff;text-transform:uppercase;
       letter-spacing:.12em;margin:16px 0 8px;border-bottom:1px solid #252b40;padding-bottom:5px}}
  .badge{{display:inline-flex;align-items:center;gap:10px;
    background:linear-gradient(135deg,#1a2545,#1d2a50);border:1px solid #3a4a80;
    border-radius:50px;padding:10px 24px;font-size:1.4rem;font-weight:600;color:#a0b4ff;
    margin-bottom:10px;box-shadow:0 0 30px #7c8fff22}}
  .dot{{width:10px;height:10px;background:#4dffb4;border-radius:50%;
    box-shadow:0 0 8px #4dffb4;animation:pulse 1.4s ease-in-out infinite;flex-shrink:0}}
  @keyframes pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.5;transform:scale(.7)}}}}
  .bar-bg{{background:#252b40;border-radius:99px;height:8px;margin:6px 0 12px;overflow:hidden}}
  .bar-fg{{height:100%;border-radius:99px;background:linear-gradient(90deg,#7c8fff,#b87cff);
    transition:width .3s ease}}
  .conf-label{{font-size:.78rem;color:#5a6177;margin-bottom:4px}}
  .metrics{{display:flex;gap:10px;margin-bottom:14px}}
  .mc{{flex:1;background:linear-gradient(135deg,#161b27,#1a2035);border:1px solid #252b40;
    border-radius:12px;padding:12px;text-align:center}}
  .mv{{font-size:1.5rem;font-weight:700;color:#7c8fff}}
  .ml{{font-size:.68rem;color:#5a6177;text-transform:uppercase;letter-spacing:.1em;margin-top:2px}}
  .hist-item{{display:flex;justify-content:space-between;align-items:center;
    padding:6px 10px;margin:3px 0;background:#13182280;border-radius:8px;
    border-left:3px solid #7c8fff;font-size:.85rem}}
  .hist-sign{{font-weight:600;color:#c8d0ff}}
  .hist-conf{{color:#5a6177;font-size:.74rem}}
  .idle{{color:#3a4060;font-size:.85rem;padding:8px 0}}
  .btn{{display:inline-block;margin-top:14px;padding:8px 20px;
    background:linear-gradient(135deg,#7c8fff,#b87cff);color:#fff;border:none;
    border-radius:9px;font-weight:600;font-size:.85rem;cursor:pointer;transition:opacity .2s}}
  .btn:hover{{opacity:.85}}
  .slider-row{{margin-bottom:10px}}
  .slider-row label{{font-size:.78rem;color:#7c8fff;display:block;margin-bottom:4px}}
  .slider-row input{{width:100%;accent-color:#7c8fff}}
  .slider-val{{font-size:.78rem;color:#a0b4ff;float:right}}
  .err{{color:#ff6b6b;padding:16px;text-align:center}}
</style>
</head>
<body>
<div class="layout">

  <!-- VIDEO -->
  <div class="col-video">
    <div class="video-wrap">
      <img id="mjpeg" src="http://localhost:{MJPEG_PORT}/video_feed"
           onerror="document.getElementById('verr').style.display='block';this.style.display='none'">
      <div id="verr" class="err" style="display:none">
        ⚠️ Could not connect to camera stream.<br>
        <small style="color:#5a6177">Make sure port {MJPEG_PORT} is free.</small>
      </div>
    </div>
  </div>

  <!-- INFO PANEL -->
  <div class="col-info">

    <!-- Settings -->
    <div class="sh">⚙️ Settings</div>
    <div class="slider-row">
      <label>Confidence Threshold <span class="slider-val" id="lbl-conf">0.70</span></label>
      <input type="range" id="sl-conf" min="0.40" max="0.99" step="0.01" value="0.70"
             oninput="document.getElementById('lbl-conf').textContent=parseFloat(this.value).toFixed(2);pushSettings()">
    </div>
    <div class="slider-row">
      <label>Prediction Stability <span class="slider-val" id="lbl-stab">4</span></label>
      <input type="range" id="sl-stab" min="2" max="8" step="1" value="4"
             oninput="document.getElementById('lbl-stab').textContent=this.value;pushSettings()">
    </div>
    <div class="slider-row">
      <label>Predict Every N Frames <span class="slider-val" id="lbl-pe">2</span></label>
      <input type="range" id="sl-pe" min="1" max="6" step="1" value="2"
             oninput="document.getElementById('lbl-pe').textContent=this.value;pushSettings()">
    </div>

    <!-- Current detection -->
    <div class="sh">🔍 Current Detection</div>
    <div id="badge-area"><p class="idle">Perform a sign gesture to begin.</p></div>

    <!-- Metrics -->
    <div class="sh">📊 Stats</div>
    <div class="metrics">
      <div class="mc"><div class="mv" id="m-total">0</div><div class="ml">Total</div></div>
      <div class="mc"><div class="mv" id="m-unique">0</div><div class="ml">Unique</div></div>
      <div class="mc"><div class="mv" id="m-avg">0%</div><div class="ml">Avg Conf</div></div>
    </div>

    <!-- Known signs -->
    <div class="sh">🗂️ Known Signs</div>
    <div style="line-height:2.2">{known_pills}</div>

    <!-- History -->
    <div class="sh">📜 History</div>
    <div id="history-list"><p class="idle">No detections yet.</p></div>
    <button class="btn" onclick="clearHistory()">🗑️ Clear History</button>

  </div>
</div>

<script>
  const PORT = {MJPEG_PORT};

  function pushSettings() {{
    fetch(`http://localhost:${{PORT}}/settings`, {{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{
        conf_threshold: parseFloat(document.getElementById('sl-conf').value),
        stability:      parseInt(document.getElementById('sl-stab').value),
        predict_every:  parseInt(document.getElementById('sl-pe').value),
      }})
    }}).catch(()=>{{}});
  }}

  function clearHistory() {{
    fetch(`http://localhost:${{PORT}}/clear`, {{method:'POST'}})
      .then(()=>pollState())
      .catch(()=>{{}});
  }}

  function fmt(ts) {{
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString();
  }}

  async function pollState() {{
    try {{
      const r   = await fetch(`http://localhost:${{PORT}}/state`);
      const s   = await r.json();

      // Badge
      const ba = document.getElementById('badge-area');
      if (s.sign) {{
        const pct = (s.conf * 100).toFixed(1);
        ba.innerHTML =
          `<div class="badge"><div class="dot"></div>${{s.sign}}</div>` +
          `<div class="conf-label">Confidence: ${{pct}}%</div>` +
          `<div class="bar-bg"><div class="bar-fg" style="width:${{pct}}%"></div></div>`;
      }} else {{
        ba.innerHTML = '<p class="idle">Perform a sign gesture to begin.</p>';
      }}

      // Metrics
      document.getElementById('m-total').textContent  = s.total;
      document.getElementById('m-unique').textContent = s.unique;
      document.getElementById('m-avg').textContent    = s.avg_conf.toFixed(0) + '%';

      // History
      const hl = document.getElementById('history-list');
      if (s.history.length) {{
        hl.innerHTML = s.history.map(e =>
          `<div class="hist-item">
            <span class="hist-sign">${{e.sign}}</span>
            <span class="hist-conf">${{(e.conf*100).toFixed(1)}}% · ${{fmt(e.ts)}}</span>
          </div>`
        ).join('');
      }} else {{
        hl.innerHTML = '<p class="idle">No detections yet.</p>';
      }}
    }} catch(e) {{}}
  }}

  // Poll every 500 ms — pure JS, zero Streamlit rerenders
  pollState();
  setInterval(pollState, 500);
</script>
</body>
</html>
""", height=780)
