import threading
from flask import Flask, Response
from flask_compress import Compress
from roboflow import Roboflow
import time
import logging
import requests  # Import requests for HTTP requests
import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls

# Initialize Flask app
app = Flask(__name__)
Compress(app)  # Enable compression for better performance

# Set up logging
logging.basicConfig(filename='wood_count.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Initialize picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 360)})  # Set resolution to HD
picam2.configure(config)

picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Roboflow API setup
API_KEY = "MGOY1rQ1SFCIi3Vi7Pse"  # Replace with your actual API key
WORKSPACE = "yolo-wood"
MODEL_ENDPOINT = "project-design-ekhku"
VERSION = 3
BACKEND = "http://localhost:YOUR_PORT"

# Initialize Roboflow model
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(MODEL_ENDPOINT)
model = project.version(VERSION).model

# Class mapping for defects
class_mapping = {
    'Blue_stain': 'Crack',
    'Crack': 'Dead Knot',
    'Death_knot': 'Knot Missing',
    'Knot_missing': 'Knot with Crack',
    'knot_with_crack': 'Live Knot',
    'Live_knot': 'Marrow',
    'Marrow': 'Quartzite',
    'overgrown': 'Resin',
}

# Global variables for predictions and counts
predictions = []
lock = threading.Lock()  # Create a lock for thread synchronization
detected_woods = {}  # To keep track of counted wood pieces and their defects
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

def predict(frame):
    """ Predicts the objects in the frame using the Roboflow model. """
    cv2.imwrite("temp_frame.jpg", frame)  # Save the frame temporarily
    prediction = model.predict("temp_frame.jpg", confidence=65, overlap=30).json()
    print("Raw Predictions:", prediction)  # Print raw predictions
    return prediction

def send_to_database(wood_data):
    """ Sends detected wood data to the Node.js backend for storage in MongoDB. """
    try:
        url = f"{BACKEND}/create-wood"
        response = requests.post(url, json=wood_data)
        if response.status_code == 200:
            print("Data sent to database successfully:", response.json())
        else:
            print("Failed to send data to database:", response.status_code, response.text)
    except Exception as e:
        print("Error sending data to database:", e)

def prediction_thread():
    global predictions, wood_count, detected_woods, ROI
    while True:
        frame = picam2.capture_array()
        frame = np.array(frame)  # Ensure frame is a NumPy array
        
        # Update ROI based on frame dimensions
        height, width = frame.shape[:2]
        ROI = calculate_roi(width, height)
        
        # Crop the frame to the ROI
        x1, y1, x2, y2 = ROI
        frame_cropped = frame[y1:y2, x1:x2]

        preds = predict(frame_cropped)

        with lock:  # Acquire lock to update predictions safely
            predictions.clear()  # Clear previous predictions
            if 'predictions' in preds:
                for pred in preds['predictions']:
                    confidence = pred['confidence']
                    object_id = pred['class']  # Use class as ID
                    
                    # Map defect name using class_mapping
                    defect_name = class_mapping.get(object_id, object_id)  # Use the mapped name or original if not found

                    print(f'Object ID: {object_id}, Confidence: {confidence}, Defect: {defect_name}')  # Debugging line

                    if confidence >= 0.65:  # Only confidence level checked
                        wood_piece_id = f"{object_id}_{int(pred['x'])}_{int(pred['y'])}"

                        if wood_piece_id not in detected_woods:
                            wood_count += 1
                            detected_woods[wood_piece_id] = {'defects': {defect_name: 1}}
                            logging.info(f'Wood counted: {wood_count} for wood piece {wood_piece_id} with defect {defect_name}')
                            print(f'Wood counted: {wood_count} for wood piece {wood_piece_id} with defect {defect_name}')  # Debugging line
                            
                            defect_data = detected_woods[wood_piece_id]['defects']
                            defectNo = sum(defect_data.values())

                            wood_data = {
                                "woodCount": wood_count,
                                "defectType": defect_name,
                                "defectNo": defectNo,
                                "woodClassification": object_id,
                                "date": time.strftime("%Y-%m-%d"),
                                "time": int(time.strftime("%H%M"))
                            }
                            send_to_database(wood_data)  # Send data to the database
                        else:
                            if defect_name in detected_woods[wood_piece_id]['defects']:
                                detected_woods[wood_piece_id]['defects'][defect_name] += 1
                            else:
                                detected_woods[wood_piece_id]['defects'][defect_name] = 1

                        adjusted_x1 = int(pred['x'] * (x2 - x1) / 640) + x1
                        adjusted_y1 = int(pred['y'] * (y2 - y1) / 360) + y1
                        adjusted_width = int(pred.get('width', 0) * (x2 - x1) / 640)
                        adjusted_height = int(pred.get('height', 0) * (y2 - y1) / 360)
                        adjusted_x2 = adjusted_x1 + adjusted_width
                        adjusted_y2 = adjusted_y1 + adjusted_height
                        
                        predictions.append({
                            'class': defect_name,  # Use the mapped defect name
                            'confidence': confidence,
                            'x1': adjusted_x1,
                            'y1': adjusted_y1,
                            'x2': adjusted_x2,
                            'y2': adjusted_y2
                        })

        time.sleep(3)

# Start the prediction thread
threading.Thread(target=prediction_thread, daemon=True).start()

def generate_frames():
    prev_time = time.time()
    while True:
        frame = picam2.capture_array()
        frame = np.array(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw the ROI on the frame
        x1, y1, x2, y2 = ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw ROI in blue

        with lock:  # Acquire lock to read predictions safely
            cv2.putText(frame, f'Wood Count: {wood_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if detected_woods:
                recent_wood_id = next(reversed(detected_woods))  # Get the most recent wood piece
                recent_data = detected_woods[recent_wood_id]
                
                defect_info = recent_data['defects']
                wood_no = wood_count

                cv2.putText(frame, f'Wood No. {wood_no}', (3, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                start_y = 70
                for defect, count in defect_info.items():
                    # Use the mapped defect name for display
                    mapped_defect = class_mapping.get(defect, defect)  # Get mapped name or original
                    cv2.putText(frame, f'Defects:', (3, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f'{mapped_defect}: {count}', (3, start_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    start_y += 15

            for pred in predictions:
                if pred['confidence'] >= 0.65:
                    # Draw bounding box
                    cv2.rectangle(frame, (pred['x1'], pred['y1']), (pred['x2'], pred['y2']), (0, 0, 255), 2)  # Red color
                    # Use the mapped class name for display
                    mapped_class_name = class_mapping.get(pred['class'], pred['class'])  # Get mapped name or original
                    cv2.putText(frame, f"{mapped_class_name} ({pred['confidence'] * 100:.1f}%)",
                                (pred['x1'], pred['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text

            fps = 1 / (time.time() - prev_time) if prev_time != time.time() else 0
            prev_time = time.time()
            cv2.putText(frame, f'FPS: {fps:.2f}', (frame.shape[1] - 100, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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