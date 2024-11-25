import threading
from flask import Flask, Response
from flask_compress import Compress
import time
import logging
import numpy as np
import cv2
from picamera2 import Picamera2, Preview

# Initialize Flask app
app = Flask(__name__)
Compress(app)  # Enable compression for better performance

# Set up logging
logging.basicConfig(filename='wood_count.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Initialize picamera2 with HD resolution
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})  # Set resolution to HD
picam2.configure(config)
picam2.start()

# Global variables
wood_count = 0  # Initialize wood count

def calculate_roi(width, height):
    """Calculate the centered ROI based on the frame dimensions."""
    roi_width = int(width * 0.7)  # 70% of the width
    roi_height = int(height * 0.5)  # 50% of the height
    x1 = (width - roi_width) // 2
    y1 = (height - roi_height) // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    return (x1, y1, x2, y2)

def prediction_thread():
    global wood_count
    while True:
        frame = picam2.capture_array()
        frame = np.array(frame)  # Ensure frame is a NumPy array

        # Update ROI based on frame dimensions
        height, width = frame.shape[:2]
        ROI = calculate_roi(width, height)

        # Crop the frame to the ROI
        x1, y1, x2, y2 = ROI
        frame_cropped = frame[y1:y2, x1:x2]

        # Here you can add any processing you want to do with the cropped frame
        # For now, it simply logs the wood count
        wood_count += 1  # Example increment for demonstration
        logging.info(f'Wood counted: {wood_count}')
        
        time.sleep(3)

# Start the prediction thread
threading.Thread(target=prediction_thread, daemon=True).start()

def generate_frames():
    prev_time = time.time()
    while True:
        frame = picam2.capture_array()
        frame = np.array(frame)

        # Convert from RGB to BGR if necessary
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Change this if your camera outputs RGB

        # Update ROI based on frame dimensions
        height, width = frame.shape[:2]
        ROI = calculate_roi(width, height)
        x1, y1, x2, y2 = ROI

        # Draw the ROI on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw ROI in blue

        # Display wood count on the frame
        cv2.putText(frame, f'Wood Count: {wood_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Calculate and display FPS
        fps = 1 / (time.time() - prev_time) if prev_time != time.time() else 0
        prev_time = time.time()
        cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 100, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=8080)

    # Cleanup
    picam2.close()  # Close the camera