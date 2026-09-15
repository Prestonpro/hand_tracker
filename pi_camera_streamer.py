import cv2
import socket
import struct
import time
from picamera2 import Picamera2

def main():
    # 1. Initialize the Pi Camera via libcamera (same backend as rpicam-still)
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    # Brief warm-up so the sensor auto-exposure can settle
    time.sleep(1)
    print("Pi Camera initialized via libcamera.")

    # 2. Start a TCP server on port 8080 to beam the video over
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 8080))
    server_socket.listen(1)

    print("Server started. Waiting for laptop to connect on port 8080...")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"Laptop connected from {addr}! Streaming video...")

            try:
                while True:
                    # Capture a frame as a NumPy array (RGB888 -> convert to BGR for OpenCV)
                    frame = picam2.capture_array()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Compress the raw image matrix into a JPEG (quality 80 is a good network balance)
                    result, frame_encoded = cv2.imencode(
                        '.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    data = frame_encoded.tobytes()

                    # We package the data by first sending the total 'length' of the JPEG chunks,
                    # then we send the actual JPEG chunks. This is so the laptop knows exactly when a frame ends!
                    client_socket.sendall(struct.pack(">L", len(data)) + data)

                    # Extremely brief delay to prevent completely flooding the network
                    time.sleep(0.01)

            except Exception as e:
                print(f"Laptop disconnected: {e}")
            finally:
                client_socket.close()
                print("Finished stream session. Waiting for laptop to reconnect...")
    finally:
        picam2.stop()

if __name__ == '__main__':
    main()
