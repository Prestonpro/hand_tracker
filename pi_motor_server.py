import socket
import json
import time
import struct
import threading

from gpiozero import AngularServo
try:
    from gpiozero.pins.pigpio import PiGPIOFactory
    _pin_factory = PiGPIOFactory()
    print("Using pigpio hardware PWM — servos will be smooth.")
except Exception as e:
    _pin_factory = None
    print(f"pigpio not available ({e}), falling back to software PWM (may jitter).")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False
    print("picamera2 not installed. Camera stream will be disabled.")

try:
    import board
    import neopixel_spi
    HAS_NEOPIXEL = True
except Exception as e:
    print(f"\n[ERROR] NeoPixel import failed! Exactly why: {e}")
    print("Adafruit NeoPixel SPI not installed. Skipping LED.")
    HAS_NEOPIXEL = False

# --- Configuration ---
# PWM Pins for Servos
SERVO_PINS = [18, 23, 24]
servos = []

# NeoPixel setup
# For Raspberry Pi 5, we MUST use SPI because the Pi 5's new RP1 chip removed native data clocking.
# You MUST wire the light's data pin to GPIO 10 (which is the SPI MOSI pin).
NUM_PIXELS = 16  # Change this to match your ring size
pixels = None

try:
    # Initialize the Servos
    for i, pin in enumerate(SERVO_PINS):
        kwargs = dict(
            initial_angle=0,
            # We widen the logical bounds slightly if you send values outside -90 to 90,
            # but note that PWM bounds (0.5ms to 2.5ms) remain standard servo bounds.
            min_angle=-180,
            max_angle=180,
            min_pulse_width=0.0005,
            max_pulse_width=0.0025,
        )
        if _pin_factory:
            kwargs['pin_factory'] = _pin_factory
        servo = AngularServo(pin, **kwargs)
        servos.append(servo)
        print(f"Servo initialized on GPIO {pin}.")
except Exception as e:
    print(f"Error initializing servos: {e}")

if HAS_NEOPIXEL:
    try:
        # Initializing SPI NeoPixels across GPIO 10
        pixels = neopixel_spi.NeoPixel_SPI(board.SPI(), NUM_PIXELS, auto_write=True)
        pixels.fill((0, 0, 0)) # Start off
        print(f"NeoPixels mapped to SPI MOSI (GPIO 10).")
    except Exception as e:
        print(f"Error initializing SPI NeoPixels: {e}")

# --- UDP Server for Servos and Lights ---
def udp_server_loop():
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening for UDP commands on port {UDP_PORT}...")
    
    while True:
        data, addr = sock.recvfrom(1024)
        print(f"[UDP] Received from {addr}: {data[:80]}")  # DEBUG
        try:
            message = json.loads(data.decode("utf-8"))
            
            # Handle motor data
            if "angles" in message:
                angles = message["angles"]
                if isinstance(angles, list):
                    DEAD_ZONE = 2.0   # degrees — ignore small changes to prevent jitter
                    for i, angle in enumerate(angles):
                        if i < len(servos):
                            current = servos[i].angle
                            current_val = current if current is not None else 0
                            delta = abs(angle - current_val)
                            if delta >= DEAD_ZONE:
                                try:
                                    servos[i].angle = angle
                                    print(f"  Motor {i+1} (GPIO {SERVO_PINS[i]}): {current_val:.1f} → {angle:.1f}")
                                except Exception as e:
                                    print(f"  Motor {i+1} (GPIO {SERVO_PINS[i]}) ERROR: {e}")
            
            # Handle neopixel data (e.g. {"light": {"on": true, "color": "#ff0000"}})
            if "light" in message and pixels is not None:
                light = message["light"]
                is_on = light.get("on", False)
                hex_color = light.get("color", "#000000")
                
                # Parse hex to RGB
                hex_color = hex_color.lstrip('#')
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    
                    if is_on:
                        pixels.fill((r, g, b))
                    else:
                        pixels.fill((0, 0, 0))

        except json.JSONDecodeError:
            pass # ignore malformed packets
# --- Camera TCP Stream Server (uses libcamera via picamera2 for IMX477) ---
def camera_server_loop():
    if not HAS_PICAMERA2 or not HAS_CV2:
        print("Camera server disabled (picamera2 or OpenCV not installed).")
        return

    try:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)  # Let auto-exposure settle
        print("Camera initialized via libcamera (picamera2).")

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 8080))
        server_socket.listen(1)
        print("Camera TCP stream ready on port 8080...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Laptop {addr} connected for live video!")

            try:
                while True:
                    frame = picam2.capture_array()  # RGB888 numpy array
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    result, frame_encoded = cv2.imencode(
                        '.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    data = frame_encoded.tobytes()
                    client_socket.sendall(struct.pack(">L", len(data)) + data)
                    time.sleep(0.01)  # tiny delay to avoid network flood
            except Exception as e:
                print(f"Laptop video disconnected: {e}")
            finally:
                client_socket.close()

    except Exception as e:
        print(f"Camera background thread error: {e}")
    finally:
        try:
            picam2.stop()
        except Exception:
            pass

if __name__ == "__main__":
    # Start the camera stream in the background
    cam_thread = threading.Thread(target=camera_server_loop, daemon=True)
    cam_thread.start()
    
    # Run the motor/light listener in the main thread
    udp_server_loop()
