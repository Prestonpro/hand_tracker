import socket
import json
import time
import math

# --- Configuration ---
# REPLACE THIS with your Raspberry Pi's actual IP address!
# To find it, open a terminal on your Pi and type: hostname -I
PI_IP_ADDRESS = "10.136.0.104" 
UDP_PORT = 5005

# Setup UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_angles_to_pi(angles):
    """
    Sends a list of 3 angles to the Raspberry Pi over UDP.
    :param angles: [motor1, motor2, motor3] — use None to keep a motor at its current position.
                   Pass a plain list of 3 floats, e.g. [0, -30, 0].
    """
    # Constrain each angle for safety
    angles = [max(-180, min(180, a)) for a in angles]

    # Create JSON message in the format pi_motor_server.py expects
    message = {"angles": angles}
    data = json.dumps(message).encode("utf-8")

    try:
        sock.sendto(data, (PI_IP_ADDRESS, UDP_PORT))
        print(f"Sent angles: {angles} to {PI_IP_ADDRESS}")
    except Exception as e:
        print(f"Failed to send data: {e}")

if __name__ == "__main__":
    print(f"Connected to {PI_IP_ADDRESS}:{UDP_PORT}")
    print("Enter 3 comma-separated angles for [M1, M2, M3] (e.g. 0,-30,0)")
    print("Type 'q' to quit.")

    try:
        # Initial reset to zero position
        send_angles_to_pi([0, 0, 0])

        while True:
            user_input = input("\nEnter angles M1,M2,M3: ").strip()

            if user_input.lower() == 'q':
                break

            try:
                parts = [float(v) for v in user_input.split(',')]
                if len(parts) != 3:
                    print("Please enter exactly 3 values separated by commas.")
                    continue
                send_angles_to_pi(parts)
            except ValueError:
                print("Please enter valid numbers (e.g. -42,-30,0).")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()
