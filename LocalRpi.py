import threading
from flask import Flask, Response
from flask_compress import Compress
import time
import logging
import numpy as np
import cv2
from picamera2 import Picamera2
import torch
from pathlib import Path

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

# Load the YOLOv5 model
model_path = '/home/inspectura/Desktop/inspecturaX-main/yolov5s_results4/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str('/home/inspectura/Desktop/inspecturaX-main/yolov5/yolov5m.pt'), force_reload=True)
print("Model loaded successfully.")

# Global variables
wood_count = 0  # Initialize wood count
predictions = []
lock = threading.Lock()  # Create a lock for thread synchronization

def calculate_roi(width, height):
    """Calculate the centered ROI based on the frame dimensions."""
    roi_width = int(width * 0.7)  # 70% of the width
    roi_height = int(height * 0.5)  # 50% of the height
    x1 = (width - roi_width) // 2
    y1 = (height - roi_height) // 2
    x2 = x1 + roi_width
    y2 = y1 + roi_height
    return (x1, y1, x2, y2)

def predict(frame):
    """Predicts the objects in the frame using the YOLOv5 model."""
    results = model(frame)  # Perform inference
    return results.pandas().xyxy[0]  # Get results as a DataFrame

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

        preds = predict(frame_cropped)  # Get predictions from the YOLOv5 model

        with lock:  # Acquire lock to update predictions safely
            predictions.clear()  # Clear previous predictions
            for _, pred in preds.iterrows():
                confidence = pred['confidence']
                if confidence >= 0.80:  # Only process predictions with high confidence
                    wood_count += 1  # Increment wood count
                    predictions.append({
                        'class': pred['name'],
                        'confidence': confidence,
                        'x1': int(pred['xmin']),
                        'y1': int(pred['ymin']),
                        'x2': int(pred['xmax']),
                        'y2': int(pred['ymax'])
                    })
                    logging.info(f'Wood counted: {wood_count} for object {pred["name"]}')

        time.sleep(3)

# Start the prediction thread
threading.Thread(target=prediction_thread, daemon=True).start()

def generate_frames():
    prev_time = time.time()
    while True:
        frame = picam2.capture_array()
        frame = np.array(frame)

        # Convert from RGB to BGR if necessary
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with lock:  # Acquire lock to read predictions safely
            for pred in predictions:
                if pred['confidence'] >= 0.80:
                    cv2.rectangle(frame, (pred['x1'], pred['y1']), (pred['x2'], pred['y2']), (0, 0, 255), 2)
                    cv2.putText(frame, f"{pred['class']} ({pred['confidence'] * 100:.1f}%)",
                                (pred['x1'], pred['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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