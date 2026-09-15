import cv2
import socket
import struct
import json
import time
import numpy as np
import mediapipe as mp
import threading

# ── Config ──────────────────────────────────────────────────────────────────
PI_IP         = "10.246.196.9"
VIDEO_PORT    = 8080          # TCP – Pi camera stream
UDP_PORT      = 5005          # UDP – motor commands
MODEL_PATH    = "hand_landmarker.task"

# Motor angle limits
MOTOR_MINS  = [-14, -7, -86]
MOTOR_MAX   = 90

# How often to send motor commands (seconds). 
SEND_INTERVAL = 1.0          # Exactly 1s per the updated kinematics requirement
# ────────────────────────────────────────────────────────────────────────────

# Hand landmark indices (MediaPipe)
WRIST       = 0
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20
INDEX_MCP   = 5
MIDDLE_MCP  = 9
RING_MCP    = 13
PINKY_MCP   = 17

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
        ax, ay = int(landmarks[a].x * w), int(landmarks[a].y * h)
        bx, by = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(image, (ax, ay), (bx, by), (0, 255, 0), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)


def remap(value, in_min, in_max, out_min, out_max):
    value = max(in_min, min(in_max, value))
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)


def landmarks_to_angles(landmarks):
    wrist = landmarks[WRIST]

    # Motor 1 (Swivel Base) – Tracks X-Axis (Left/Right)
    # The user noted it was inverted horizontally.
    angle1 = remap(wrist.x, 0.0, 1.0, -90, 90)

    # Motor 2 (Shoulder) – Tracks hand height / depth
    # As the hand gets larger (closer to camera / higher Z), motor rotates up.
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    avg_dist = np.mean([
        np.hypot(landmarks[t].x - wrist.x, landmarks[t].y - wrist.y)
        for t in tips
    ])
    angle2 = remap(avg_dist, 0.10, 0.40, 0, -90)

    # Motor 3 (Elbow) – Constrained to parallel forearm compensation
    # Base defaults: angle2 = -30, angle3 = 0. Offset is exactly + 30.0
    angle3 = angle2 + 30.0

    return [round(angle1, 1), round(angle2, 1), round(angle3, 1)]


def receive_frame(client_socket, buffer, payload_size):
    while len(buffer) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            raise ConnectionError("Pi closed the stream.")
        buffer += packet

    msg_size = struct.unpack(">L", buffer[:payload_size])[0]
    buffer = buffer[payload_size:]

    while len(buffer) < msg_size:
        buffer += client_socket.recv(4096)

    frame_data = buffer[:msg_size]
    buffer = buffer[msg_size:]

    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame, buffer

class VideoStreamer:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.frame = None
        self.frame_ts = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        payload_size = struct.calcsize(">L")
        while self.running:
            tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_sock.settimeout(5.0)
            try:
                tcp_sock.connect((self.ip, self.port))
                tcp_sock.settimeout(None)
                buffer = b""
                while self.running:
                    f, buffer = receive_frame(tcp_sock, buffer, payload_size)
                    if f is not None:
                        self.frame = f
                        self.frame_ts = time.perf_counter()
            except Exception as e:
                pass
            finally:
                tcp_sock.close()
                time.sleep(1)


def main():
    # ── UDP socket for sending motor commands ───────────────────────────────
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ── MediaPipe hand landmarker ────────────────────────────────────────────
    BaseOptions        = mp.tasks.BaseOptions
    HandLandmarker     = mp.tasks.vision.HandLandmarker
    HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
    RunningMode        = mp.tasks.vision.RunningMode

    # Restoring VIDEO mode for much higher tracking stability across frames
    options = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
    )

    print("Loading MediaPipe model...")
    with HandLandmarker.create_from_options(options) as landmarker:
        
        print(f"Starting Video Background Thread pointing at {PI_IP}:{VIDEO_PORT}...")
        stream = VideoStreamer(PI_IP, VIDEO_PORT)
        
        print("Waiting for physical camera connection...")
        while stream.frame is None:
            time.sleep(0.1)
            
        print("Connected! Running hand tracker constraint loop. Press 'q' to quit.")
        
        last_send         = 0.0
        pTime             = time.perf_counter()
        angles            = [0, 0, 0]  # Default startup position
        last_processed_ts = 0.0
        light_is_on       = False

        while True:
            # Sync constraint: We must only run detection if the frame is fully refreshed strictly newer than previous
            if stream.frame_ts is None or stream.frame_ts <= last_processed_ts:
                time.sleep(0.005)
                continue

            # Drop older frames, pull exactly the latest one in RAM right now
            frame = stream.frame.copy()
            current_ts = stream.frame_ts
            last_processed_ts = current_ts
            
            # ── Hand tracking ────────────────────────────────────────
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
            # Use VIDEO mode to utilize previous frame state and vastly reduce loss
            result = landmarker.detect_for_video(mp_img, int(current_ts * 1000))

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, landmarks)

                # ── Send motor angles at throttled rate ──────────────
                now = time.perf_counter()
                
                # Fire immediately if hand just appeared, otherwise respect throttle
                if not light_is_on or (now - last_send >= SEND_INTERVAL):
                    angles  = landmarks_to_angles(landmarks)
                    payload = json.dumps({
                        "angles": angles,
                        "light": {"on": True, "color": "#ffffff"}
                    }).encode("utf-8")
                    try:
                        udp_sock.sendto(payload, (PI_IP, UDP_PORT))
                    except:
                        pass
                    last_send = now
                    light_is_on = True

                # Re-calculate avg_dist strictly for UI overlay
                avg_dist = np.mean([
                    np.hypot(landmarks[t].x - landmarks[0].x, 
                             landmarks[t].y - landmarks[0].y)
                    for t in [8, 12, 16, 20]
                ])

                cv2.putText(frame,
                    f"M1(vis):{landmarks[0].x:.2f} M3(vis):{avg_dist:.2f}",
                    (10, 110), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
            else:
                if light_is_on:
                    # Instantly turn off the physical camera NeoPixel light when lock is broken
                    payload = json.dumps({"light": {"on": False}}).encode("utf-8")
                    try:
                        udp_sock.sendto(payload, (PI_IP, UDP_PORT))
                    except:
                        pass
                    light_is_on = False
                    
                cv2.putText(frame, "No hand detected",
                    (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 2)
                    
            cv2.putText(frame,
                f"Ang1:{angles[0]:.0f} Ang2:{angles[1]:.0f} Ang3:{angles[2]:.0f}",
                (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            # ── FPS counter ──────────────────────────────────────────
            cTime = time.perf_counter()
            fps   = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime
            cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow("Pi Camera – Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stream.running = False
                break
                
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
