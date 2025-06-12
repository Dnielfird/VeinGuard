import subprocess
import numpy as np
import cv2

def capture_frame():
    temp_path = "/tmp/palm_frame.jpg"
    try:
        # Capture with minimal delay for speed
        subprocess.run([
            "libcamera-still", 
            "-n",             # no preview window
            "-t", "1",        # capture immediately
            "--width", "224", 
            "--height", "224", 
            "-o", temp_path
        ], check=True)
        frame = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            return cv2.resize(frame, (224, 224))
        return None
    except Exception as e:
        print(f"[Camera Error] {e}")
        return None
