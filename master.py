import socket
import json
import struct
import time
import threading
import asyncio
import os
import base64
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] OpenCV not found. Camera stream disabled.")

try:
    import mediapipe as mp
    HAS_MP = True
except ImportError:
    HAS_MP = False
    print("[WARN] MediaPipe not found. Hand tracking disabled.")

# ── Configuration ─────────────────────────────────────────────────────────────
PI_IP         = "10.136.0.104"
VIDEO_PORT    = 8080
UDP_PORT      = 5005
MODEL_PATH    = "hand_landmarker.task"
MOTOR_MINS    = [-14, -7, -86]
MOTOR_MAX     = 90
SEND_INTERVAL = 0.05   # seconds between auto UDP sends in hand-track mode

# ── Shared state (thread-safe) ────────────────────────────────────────────────
_lock            = threading.Lock()
_latest_b64      = None     # str: base64-encoded annotated JPEG for WebSocket push
_hand_track_mode = False    # True = motors driven by hand
_current_angles  = None     # [m1, m2, m3] or None if no hand detected

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ── Hand tracking helpers ─────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_landmarks(image, landmarks):
    h, w, _ = image.shape
    for a, b in HAND_CONNECTIONS:
        cv2.line(image,
                 (int(landmarks[a].x * w), int(landmarks[a].y * h)),
                 (int(landmarks[b].x * w), int(landmarks[b].y * h)),
                 (0, 255, 0), 2)
    for lm in landmarks:
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), cv2.FILLED)

def remap(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(in_max, value))
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)

def landmarks_to_angles(landmarks):
    wrist = landmarks[0]
    angle1 = remap(wrist.x, 0.0, 1.0, MOTOR_MAX, MOTOR_MINS[0])
    angle2 = remap(wrist.y, 0.0, 1.0, MOTOR_MAX, MOTOR_MINS[1])
    tips = [8, 12, 16, 20]
    avg_dist = float(np.mean([
        np.hypot(landmarks[t].x - wrist.x, landmarks[t].y - wrist.y)
        for t in tips
    ]))
    angle3 = remap(avg_dist, 0.10, 0.40, MOTOR_MINS[2], MOTOR_MAX)
    return [round(angle1, 1), round(angle2, 1), round(angle3, 1)]

def receive_frame(sock, buf, payload_size):
    while len(buf) < payload_size:
        packet = sock.recv(4096)
        if not packet:
            raise ConnectionError("Pi closed the stream.")
        buf += packet
    msg_size = struct.unpack(">L", buf[:payload_size])[0]
    buf = buf[payload_size:]
    while len(buf) < msg_size:
        buf += sock.recv(4096)
    frame = cv2.imdecode(np.frombuffer(buf[:msg_size], dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame, buf[msg_size:]

# ── Background thread: Pi camera + hand detection ─────────────────────────────
def camera_thread():
    global _latest_b64, _current_angles

    if not HAS_CV2 or not HAS_MP:
        print("[Camera] Disabled — OpenCV or MediaPipe missing.")
        return

    BaseOptions    = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HLOptions      = mp.tasks.vision.HandLandmarkerOptions
    RunningMode    = mp.tasks.vision.RunningMode

    # Use IMAGE mode (stateless) — same as the working hackathon server
    options = HLOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,   # stateless, no timestamp required
        num_hands=1,
    )

    payload_size = struct.calcsize(">L")

    with HandLandmarker.create_from_options(options) as landmarker:
        last_send = 0.0

        while True:
            print(f"[Camera] Connecting to {PI_IP}:{VIDEO_PORT}...")
            tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                tcp.connect((PI_IP, VIDEO_PORT))
                print("[Camera] Connected. Streaming.")
            except Exception as e:
                print(f"[Camera] {e} — retry in 5s")
                tcp.close()
                time.sleep(5)
                continue

            buf = b""
            try:
                while True:
                    frame, buf = receive_frame(tcp, buf, payload_size)
                    if frame is None:
                        continue

                    # Run hand tracking (IMAGE mode — stateless per-frame)
                    try:
                        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
                        result = landmarker.detect(mp_img)   # stateless detect
                    except RuntimeError:
                        return  # interpreter shutting down
                    except Exception as e:
                        print(f"[Camera] MediaPipe error (skipping): {e}")
                        continue

                    if result.hand_landmarks:
                        lms    = result.hand_landmarks[0]
                        angles = landmarks_to_angles(lms)
                        draw_landmarks(frame, lms)

                        with _lock:
                            _current_angles = angles

                        # Auto-send UDP when hand tracking mode is active
                        now = time.perf_counter()
                        with _lock:
                            tracking = _hand_track_mode
                        if tracking and now - last_send >= SEND_INTERVAL:
                            constrained = [
                                max(MOTOR_MINS[i], min(MOTOR_MAX, angles[i]))
                                for i in range(3)
                            ]
                            udp_sock.sendto(
                                json.dumps({"angles": constrained}).encode(),
                                (PI_IP, UDP_PORT)
                            )
                            last_send = now

                        cv2.putText(frame,
                            f"M1:{angles[0]:.0f}  M2:{angles[1]:.0f}  M3:{angles[2]:.0f}",
                            (8, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 0), 2)
                    else:
                        with _lock:
                            _current_angles = None
                        cv2.putText(frame, "No hand detected",
                            (8, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.4, (80, 120, 255), 2)

                    # Mode badge (top-left)
                    with _lock:
                        mode = _hand_track_mode
                    label = "HAND TRACK" if mode else "MANUAL"
                    color = (0, 230, 120) if mode else (160, 160, 255)
                    cv2.putText(frame, label, (8, 28),
                        cv2.FONT_HERSHEY_PLAIN, 1.8, color, 2)

                    # Encode as JPEG then base64 — exactly like the hackathon server
                    _, buf_jpg = cv2.imencode(".jpg", frame,
                                             [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    b64 = base64.b64encode(buf_jpg).decode("utf-8")
                    with _lock:
                        _latest_b64 = b64

            except Exception as e:
                print(f"[Camera] Lost: {e} — retry in 5s")
            finally:
                tcp.close()
                time.sleep(5)


# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI()

class AnglesModel(BaseModel):
    angles: list[float]

class LightModel(BaseModel):
    on: bool
    color: str

class ModeModel(BaseModel):
    hand_track: bool

@app.post("/api/move")
async def move_motors(data: AnglesModel):
    constrained = [
        max(MOTOR_MINS[i] if i < len(MOTOR_MINS) else -90, min(MOTOR_MAX, a))
        for i, a in enumerate(data.angles)
    ]
    try:
        udp_sock.sendto(json.dumps({"angles": constrained}).encode(), (PI_IP, UDP_PORT))
        return {"status": "success", "angles": constrained}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/light")
async def set_light(data: LightModel):
    try:
        udp_sock.sendto(
            json.dumps({"light": {"on": data.on, "color": data.color}}).encode(),
            (PI_IP, UDP_PORT)
        )
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/mode")
async def set_mode(data: ModeModel):
    global _hand_track_mode
    with _lock:
        _hand_track_mode = data.hand_track
    print(f"[Mode] hand_track={'ON' if data.hand_track else 'OFF'}")
    return {"status": "success", "hand_track": data.hand_track}

@app.get("/api/status")
async def get_status():
    with _lock:
        mode   = _hand_track_mode
        angles = _current_angles
    return {
        "hand_track":    mode,
        "hand_detected": angles is not None,
        "angles":        angles,
    }

# ── WebSocket video stream (base64 JPEG push, ~20fps) ─────────────────────────
@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected for video")
    try:
        while True:
            with _lock:
                b64 = _latest_b64
            if b64 is not None:
                await websocket.send_json({"frame": b64})
            await asyncio.sleep(0.05)   # 20 fps push
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        print("[WS] Client disconnected")

@app.get("/api/config")
async def get_config():
    return {"pi_ip": PI_IP}

# Static files last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")

    print("=" * 50)
    print("  IRIS Master Server")
    print(f"  Web UI  -> http://127.0.0.1:8000")
    print(f"  Pi UDP  -> {PI_IP}:{UDP_PORT}")
    print(f"  Pi CAM  -> {PI_IP}:{VIDEO_PORT}")
    print("=" * 50)

    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    uvicorn.run("master:app", host="127.0.0.1", port=8000, reload=False)
