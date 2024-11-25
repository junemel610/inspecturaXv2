import os
import time
import subprocess
import json

# Start the Flask app
flask_process = subprocess.Popen(['python', 'app.py'])

# Give the Flask app some time to start
time.sleep(2)

# Start Ngrok
ngrok_process = subprocess.Popen(['C:\\Users\\Meljune\\ngrok\\ngrok.exe', 'http', '8080'])

print("Ngrok is running...")

# Wait for Ngrok to initialize
time.sleep(2)
try:
    # Keep the script running while the Flask app and Ngrok are running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Terminate the processes on exit
    flask_process.terminate()
    ngrok_process.terminate()