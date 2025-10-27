# app.py
import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
from collections import deque
import os

st.set_page_config(page_title="VisionGuard AI", layout="wide")

# Simple hacker-style CSS
st.markdown("""
<style>
body { background-color:#020306; color:#bfeee0 }
.title { font-size:32px; color:#00ffd5; font-weight:700; text-shadow:0 0 8px #00ffd5; }
.box { background:rgba(255,255,255,0.02); padding:10px; border-radius:10px; }
.alert { padding:8px; margin-bottom:6px; border-radius:8px; background:#0f2330; box-shadow:0 0 8px #04303a; }
.alert.high { background:#3a0808; color:#ffdede }
.kv { font-family:monospace }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>VisionGuard — AI Safety Detection</div>", unsafe_allow_html=True)
st.markdown("Hacker-style UI • Live webcam • Specific alerts (no helmet, phone, headphones, rash driving)")

# Sidebar controls
st.sidebar.header("Controls")
source_mode = st.sidebar.selectbox("Camera source", ("Webcam (local)", "Uploaded video"))
cam_index = st.sidebar.number_input("Webcam index", min_value=0, max_value=4, value=0, step=1)
uploaded_file = st.sidebar.file_uploader("Upload video (mp4/avi) — only used if uploaded mode", type=["mp4", "avi", "mov"])
model_path = st.sidebar.text_input("YOLO model path", value="yolov8n.pt")
run_fps = st.sidebar.slider("Max processing FPS", min_value=1, max_value=15, value=6)
beep_checkbox = st.sidebar.checkbox("Play beep on new alerts (requires beep1.mp3)", value=False)
st.sidebar.markdown("---")
start_btn = st.sidebar.button("Start")
stop_btn = st.sidebar.button("Stop")
pause_btn = st.sidebar.button("Pause / Resume")

# placeholders
col1, col2 = st.columns([2,1])
video_placeholder = col1.empty()
alerts_placeholder = col2.empty()
counts_placeholder = col2.empty()

# session state initialization
if "running" not in st.session_state:
    st.session_state.running = False
if "paused" not in st.session_state:
    st.session_state.paused = False
if "alerts" not in st.session_state:
    st.session_state.alerts = deque(maxlen=5)   # show last 5
if "last_alert_type" not in st.session_state:
    st.session_state.last_alert_type = None
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = {}
if "counts" not in st.session_state:
    st.session_state.counts = {"helmet":0, "phone":0, "headphone":0, "rash":0}
if "prev_positions" not in st.session_state:
    st.session_state.prev_positions = {}  # for rash driving (keyed by object id/name)
if "cap" not in st.session_state:
    st.session_state.cap = None
if "model" not in st.session_state:
    st.session_state.model = None

# cooldowns (seconds) per alert type to allow re-showing after some time if repeated
ALERT_COOLDOWNS = {
    "no_helmet": 3.0,
    "phone": 3.0,
    "headphone": 3.0,
    "rash": 5.0,
    "no_detection": 1.5,
}

def safe_beep():
    if not beep_checkbox:
        return
    if os.path.exists("beep1.mp3"):
        try:
            if os.name == "posix":
                os.system("afplay beep1.mp3 >/dev/null 2>&1 &")
            else:
                os.system("start /min beep1.mp3")
        except Exception:
            pass

def push_alert(alert_type, readable_text):
    """
    Append alert only if it's not the same type as the last alert (prevents immediate consecutive duplicates),
    OR if the cooldown for that alert_type has elapsed.
    """
    now = time.time()
    last_type = st.session_state.last_alert_type
    last_time = st.session_state.last_alert_time.get(alert_type, 0)
    cooldown = ALERT_COOLDOWNS.get(alert_type, 2.0)

    # If the last alert type is the same as the new one, prevent immediate consecutive duplicates
    if alert_type == last_type:
        # allow again only after cooldown
        if now - last_time < cooldown:
            return False

    # Append alert (prepend so newest is on top)
    ts = time.strftime("%H:%M:%S")
    st.session_state.alerts.appendleft(f"{ts} — {readable_text}")
    st.session_state.last_alert_type = alert_type
    st.session_state.last_alert_time[alert_type] = now
    safe_beep()
    # update counts when appropriate
    if alert_type == "no_helmet":
        st.session_state.counts["helmet"] += 1
    elif alert_type == "phone":
        st.session_state.counts["phone"] += 1
    elif alert_type == "headphone":
        st.session_state.counts["headphone"] += 1
    elif alert_type == "rash":
        st.session_state.counts["rash"] += 1
    return True

def draw_alerts_and_counts():
    # Alerts column
    with alerts_placeholder.container():
        st.markdown("<div class='box'><h4>🔔 Live Alerts</h4>", unsafe_allow_html=True)
        for a in list(st.session_state.alerts):
            # mark high severity if contains keywords
            cls = "alert"
            if "NO HELMET" in a or "RASH DRIVING" in a:
                st.markdown(f"<div class='{cls} high'>{a}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='{cls}'>{a}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # Counts
    with counts_placeholder.container():
        st.markdown("<div class='box'><h4>📊 Violation Counts</h4>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>NO HELMET: {st.session_state.counts['helmet']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>PHONE: {st.session_state.counts['phone']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>HEADPHONES: {st.session_state.counts['headphone']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'>RASH DRIVING: {st.session_state.counts['rash']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# start/stop/pause handling
if start_btn:
    st.session_state.running = True
    st.session_state.paused = False
    # (lazy) load model
    try:
        st.session_state.model = YOLO(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.session_state.running = False

if stop_btn:
    st.session_state.running = False
    # release capture if exists
    if st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

if pause_btn:
    st.session_state.paused = not st.session_state.paused

# initialize capture if starting
if st.session_state.running and st.session_state.cap is None:
    if source_mode == "Webcam (local)":
        st.session_state.cap = cv2.VideoCapture(int(cam_index))
    else:
        if uploaded_file is not None:
            tmp_path = f"/tmp/{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            st.session_state.cap = cv2.VideoCapture(tmp_path)
        else:
            st.error("Upload a video file or switch to webcam.")
            st.session_state.running = False

# main loop
if st.session_state.running:
    cap = st.session_state.cap
    model = st.session_state.model
    if cap is None or model is None:
        st.error("Camera or model not initialized.")
    else:
        target_period = 1.0 / max(1, int(run_fps))
        last_no_detection_time = st.session_state.last_alert_time.get("no_detection", 0)

        while st.session_state.running:
            if st.session_state.paused:
                time.sleep(0.2)
                continue

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                push_alert("no_detection", "NO DETECTION — no frame / video ended")
                st.session_state.running = False
                break

            # resize for speed
            h, w = frame.shape[:2]
            scale = 640.0 / max(w, h)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            # run YOLO
            results = model(frame)

            detected_objects = []
            # parse detections
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                # boxes.xyxy and boxes.cls are tensors/arrays
                xy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
                cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls).astype(int)
                for i, box in enumerate(xy):
                    x1, y1, x2, y2 = map(int, box[:4])
                    cls_id = int(cls_ids[i])
                    label = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)
                    detected_objects.append(label.lower())

                    # draw box & label on frame
                    if label.lower() == "helmet":
                        color = (0,255,0)
                    elif "phone" in label.lower() or "cell" in label.lower():
                        color = (0,140,255)
                    elif "headphone" in label.lower() or "ear" in label.lower():
                        color = (255,0,0)
                    else:
                        color = (255,255,255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # for rash driving: store center of vehicles (we key by class+index fallback)
                    if label.lower() in ["car", "motorbike", "bus", "truck"]:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        key = f"{label.lower()}_{i}"
                        prev = st.session_state.prev_positions.get(key)
                        if prev is not None:
                            prev_x, prev_y, prev_t = prev
                            pixel_dist = math.hypot(cx - prev_x, cy - prev_y)
                            dt = max(0.001, time.time() - prev_t)
                            # heuristic convert pixel movement into km/h (dataset & camera dependent)
                            kmph = (pixel_dist * 0.05) / (dt / 3600.0)
                            if kmph > 65:
                                push_alert("rash", f"RASH DRIVING detected ({int(kmph)} km/h)")
                        st.session_state.prev_positions[key] = (cx, cy, time.time())

            # Specific alert logic from detected_objects:
            # vehicle present but no helmet
            vehicles_present = any(x in detected_objects for x in ["motorbike", "car", "person", "bus", "truck"])
            helmet_present = any("helmet" in x for x in detected_objects)

            # check phone/headphone presence
            phone_present = any("phone" in x or "cell" in x for x in detected_objects)
            headphone_present = any("headphone" in x or "ear" in x for x in detected_objects)

            # push alerts based on presence, preventing consecutive duplicates by push_alert()
            if vehicles_present and not helmet_present:
                push_alert("no_helmet", "NO HELMET — rider/driver not wearing helmet")
            if phone_present:
                push_alert("phone", "PHONE DETECTED — smartphone in use")
            if headphone_present:
                push_alert("headphone", "HEADPHONES DETECTED — audio device detected")

            # if nothing important was detected, add a periodic NO DETECTION message (but not repeating consecutively)
            important_flags = vehicles_present or helmet_present or phone_present or headphone_present
            now = time.time()
            if not important_flags:
                if now - st.session_state.last_alert_time.get("no_detection", 0) > ALERT_COOLDOWNS["no_detection"]:
                    push_alert("no_detection", "NO DETECTION")
                    st.session_state.last_alert_time["no_detection"] = now

            # show frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # draw alerts and counts
            draw_alerts_and_counts()

            # FPS control + UI responsiveness
            elapsed = time.time() - t0
            sleep_for = max(0, target_period - elapsed)
            time.sleep(sleep_for)

            # check stop button (Streamlit buttons re-run; check session flag)
            if not st.session_state.running:
                break

        # cleanup
        try:
            cap.release()
        except Exception:
            pass
        st.session_state.cap = None

# initial draw
draw_alerts_and_counts()