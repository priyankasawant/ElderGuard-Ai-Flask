from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)

# Live webcam capture
#camera = cv2.VideoCapture(0)

# Global alert status
current_status = {"status": "Normal"}

# MJPEG streaming generator
def gen_frames():
    while True:
        success, frame = ""
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream to browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route: Video stream
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route: Get current alert status
@app.route('/status')
def status():
    return jsonify(current_status)

# Route: Trigger alert from ML model
@app.route('/trigger-alert', methods=['POST'])
def trigger_alert():
    current_status["status"] = "🚨 Fall/Hit Detected"
    return jsonify({"msg": "Alert triggered"})

# Route: Reset alert
@app.route('/reset', methods=['POST'])
def reset_alert():
    current_status["status"] = "Normal"
    return jsonify({"msg": "Alert reset"})

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
