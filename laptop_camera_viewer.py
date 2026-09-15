import cv2
import socket
import struct
import numpy as np

# Replace this with your Raspberry Pi's IP address (it's loaded with your current one from the logs)
PI_IP = '10.136.0.104'
PORT = 8080

def main():
    print(f"Connecting to Raspberry Pi at {PI_IP}:{PORT}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((PI_IP, PORT))
    print("Connected successfully! Receiving video...")

    data = b""
    payload_size = struct.calcsize(">L")

    try:
        while True:
            # 1. Grab the packet size containing the next frame
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet: break
                data += packet
                
            if not data: break
                
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            
            # 2. Receive the actual JPEG chunks until we have the full image
            while len(data) < msg_size:
                data += client_socket.recv(4096)
                
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # 3. Decode the JPEG back into a matrix format and display it
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('Pi Camera Feed (from network)', frame)
            
            # Press 'q' to safely exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Connection dropped: {e}")
    finally:
        print("Closing viewer.")
        cv2.destroyAllWindows()
        client_socket.close()

if __name__ == '__main__':
    main()
