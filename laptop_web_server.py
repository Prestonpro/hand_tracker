import socket
import json
import struct
import time
import threading
import os
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MP = True
except ImportError:
    HAS_MP = False

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# --- Configuration ---
PI_IP_ADDRESS = "10.246.196.9"
UDP_PORT = 5005
VIDEO_PORT = 8080
MODEL_PATH = "hand_landmarker.task"
SEND_INTERVAL = 1.0  # Limit UDP to exactly 1 update per second

# --- Shared State ---
_autonomous_mode = False
_current_angles = None
_lock = threading.Lock()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
app = FastAPI()

# --- Hand Tracking Helpers ---
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
    
    # Motor 1 (Base Rotation)
    angle1 = 0.0
    
    # Motor 2 (Y-Axis / Arm Pitch)
    tips = [8, 12, 16, 20]
    avg_dist = float(np.mean([
        np.hypot(landmarks[t].x - wrist.x, landmarks[t].y - wrist.y)
        for t in tips
    ]))
    # Spread is approx 0.10 (fist) to 0.40 (open).
    angle2 = remap(avg_dist, 0.10, 0.40, 0, -90)
    
    # Motor 3 (Z-Axis / Left-Right + Elbow height compensation)
    base_angle3 = remap(wrist.x, 0.0, 1.0, 90, -90)
    angle3 = base_angle3 - angle2
    
    return [round(angle1, 1), round(angle2, 1), round(angle3, 1)]

def receive_frame(tcp_sock, buf, payload_size):
    while len(buf) < payload_size:
        packet = tcp_sock.recv(4096)
        if not packet:
            raise ConnectionError("Pi closed the stream.")
        buf += packet
    msg_size = struct.unpack(">L", buf[:payload_size])[0]
    buf = buf[payload_size:]
    while len(buf) < msg_size:
        buf += tcp_sock.recv(4096)
    frame = cv2.imdecode(np.frombuffer(buf[:msg_size], dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame, buf[msg_size:]


# --- Background Worker ---
def autonomous_tracking_worker():
    global _current_angles

    if not HAS_CV2 or not HAS_MP:
        print("[Tracking] Disabled — OpenCV or MediaPipe missing.")
        return

    BaseOptions    = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HLOptions      = mp.tasks.vision.HandLandmarkerOptions
    RunningMode    = mp.tasks.vision.RunningMode

    options = HLOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
    )

    payload_size = struct.calcsize(">L")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            # Check if autonomous mode is active
            with _lock:
                active = _autonomous_mode
            
            if not active:
                time.sleep(0.5)
                continue
                
            print(f"[Tracking] Autonomous Mode ON. Connecting to {PI_IP_ADDRESS}:{VIDEO_PORT}...")
            tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp.settimeout(5.0)  # Avoid indefinite hang
            try:
                tcp.connect((PI_IP_ADDRESS, VIDEO_PORT))
                tcp.settimeout(None)
                print("[Tracking] Connected. Running hand tracker.")
            except Exception as e:
                print(f"[Tracking] Could not connect: {e} — retry in 3s")
                tcp.close()
                time.sleep(3)
                continue

            buf = b""
            last_send = 0.0
            
            try:
                cv2.namedWindow("Autonomous Tracker", cv2.WINDOW_NORMAL)
                while True:
                    with _lock:
                        if not _autonomous_mode:
                            print("[Tracking] Autonomous Mode OFF. Stopping stream.")
                            break
                            
                    frame, buf = receive_frame(tcp, buf, payload_size)
                    if frame is None:
                        continue

                    # Run hand tracking
                    try:
                        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
                        result = landmarker.detect(mp_img)
                    except Exception as e:
                        print(f"[Tracking] MediaPipe error: {e}")
                        continue

                    if result.hand_landmarks:
                        lms = result.hand_landmarks[0]
                        angles = landmarks_to_angles(lms)
                        draw_landmarks(frame, lms)
                        
                        # Add text displaying angles like before
                        visual_m1_x = lms[0].x
                        visual_m3_spread = float(np.mean([np.hypot(lms[t].x - lms[0].x, lms[t].y - lms[0].y) for t in [8, 12, 16, 20]]))
                        
                        cv2.putText(frame,
                            f"M1(vis):{visual_m1_x:.2f} M3(vis):{visual_m3_spread:.2f}",
                            (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
                        cv2.putText(frame,
                            f"Ang1:{angles[0]:.0f} Ang2:{angles[1]:.0f} Ang3:{angles[2]:.0f}",
                            (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

                        with _lock:
                            _current_angles = angles

                        now = time.perf_counter()
                        if now - last_send >= SEND_INTERVAL:
                            # Send computed angles to server automatically
                            msg = {"angles": angles}
                            payload = json.dumps(msg).encode("utf-8")
                            try:
                                sock.sendto(payload, (PI_IP_ADDRESS, UDP_PORT))
                            except Exception:
                                pass
                            last_send = now
                    else:
                        with _lock:
                            _current_angles = None
                        cv2.putText(frame, "No hand detected", (10, 40), 
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                    # Show on screen as requested
                    cv2.imshow("Autonomous Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        # If user manually closes window, break but don't exit script
                        # (It'll reopen automatically if mode is still true, which is fine)
                        pass
                        
            except Exception as e:
                print(f"[Tracking] Lost stream: {e}")
            finally:
                cv2.destroyAllWindows()
                tcp.close()
                
                with _lock:
                    _current_angles = None
                time.sleep(1)


# --- REST API Endpoints ---
class AnglesModel(BaseModel):
    angles: list[float]

class LightModel(BaseModel):
    on: bool
    color: str

class ModeModel(BaseModel):
    hand_track: bool

@app.post("/api/move")
async def move_motors(data: AnglesModel):
    with _lock:
        if _autonomous_mode:
            # Ignore manual slider moves if autonomous is controlling it
            return {"status": "error", "message": "Autonomous mode active"}
            
    # Pass angles through directly — no clamping, no inversion (as requested earlier)
    message = {"angles": data.angles}
    payload = json.dumps(message).encode("utf-8")

    try:
        sock.sendto(payload, (PI_IP_ADDRESS, UDP_PORT))
        print(f"Sent manual angles: {data.angles} to {PI_IP_ADDRESS}")
        return {"status": "success", "angles": data.angles}
    except Exception as e:
        print(f"Failed to send data: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/light")
async def set_light(data: LightModel):
    message = {"light": {"on": data.on, "color": data.color}}
    payload = json.dumps(message).encode("utf-8")
    
    try:
        sock.sendto(payload, (PI_IP_ADDRESS, UDP_PORT))
        print(f"Sent light config: {message} to {PI_IP_ADDRESS}")
        return {"status": "success", "light": message["light"]}
    except Exception as e:
        print(f"Failed to send light data: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/mode")
async def set_mode(data: ModeModel):
    global _autonomous_mode
    with _lock:
        _autonomous_mode = data.hand_track
    print(f"Autonomous Mode changed to: {'ON' if _autonomous_mode else 'OFF'}")
    return {"status": "success", "mode": "hand_track" if data.hand_track else "manual"}

@app.get("/api/status")
async def get_status():
    with _lock:
        mode = _autonomous_mode
        angles = _current_angles
    return {
        "status": "success", 
        "hand_track": mode, 
        "hand_detected": angles is not None, 
        "angles": angles
    }

@app.get("/api/config")
async def get_config():
    """Provides the frontend with the Pi's IP address so it can connect to the video stream directly."""
    return {"pi_ip": PI_IP_ADDRESS}

@app.websocket("/ws/video")
async def no_camera_ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.close(1001)   # Going Away

# Mount static files last so API/WS routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
        print("Created 'static' directory. Please place your HTML/CSS/JS files there.")

    # Start the background worker for MediaPipe
    t = threading.Thread(target=autonomous_tracking_worker, daemon=True)
    t.start()

    print(f"Starting web server at http://127.0.0.1:8000")
    print(f"Will send UDP commands to {PI_IP_ADDRESS}:{UDP_PORT}")
    print("Open http://127.0.0.1:8000 in your browser to use the IRIS control panel.")

    uvicorn.run("laptop_web_server:app", host="127.0.0.1", port=8000, reload=False)
