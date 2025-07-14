import base64
from flask import Flask, request, jsonify,send_file, make_response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import math
import cvzone
import tempfile
import os

app = Flask(__name__)
CORS(app,expose_headers=["X-Prediction"])
model = YOLO("yolov8s.pt")

# Load class names
classnames = []
file = open('./classes.txt', 'r')
for line in file:
    classnames.append(line.strip())
file.close()

# Position and angle tracking
position_history = {}
angle_history = {}

# Thresholds for fall detection
velocity_threshold = 20  
angle_change_threshold = 45  
aspect_ratio_threshold = 1.5  
alert_triggered = False

@app.route("/analyze", methods=["POST"])
def analyze():
    if "frame" not in request.files:
        return {"error": "No frame"}, 400

    img = Image.open(request.files["frame"].stream).convert("RGB")
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    prediction = "Normal"

    results = model(frame)
    for r in results:
        for box in r.boxes:
           x1, y1, x2, y2 = map(int, box.xyxy[0])
           confidence = box.conf[0]
           class_detect = int(box.cls[0])
           class_detect = classnames[class_detect]
           conf = math.ceil(confidence * 100)

           height = y2 - y1
           width = x2 - x1
           aspect_ratio = width / height
           center_y = (y1 + y2) // 2

           angle = math.degrees(math.atan2(height, width))

            # Generate a unique ID for tracking
           person_id = f'{x1}{y1}{x2}_{y2}'
           if person_id not in position_history:
                position_history[person_id] = []  
           if person_id not in angle_history:
                angle_history[person_id] = []
           position_history[person_id].append(center_y)
           angle_history[person_id].append(angle)

            # Keep only the last two values for velocity and angle calculations
           if len(position_history[person_id]) > 2:
                position_history[person_id] = position_history[person_id][-2:]
           if len(angle_history[person_id]) > 2:
                angle_history[person_id] = angle_history[person_id][-2:]     

            # Calculate velocity and angle change
           velocity = position_history[person_id][-1] - position_history[person_id][-2] if len(position_history[person_id]) >= 2 else 0
           angle_change = abs(angle_history[person_id][-1] - angle_history[person_id][-2]) if len(angle_history[person_id]) >= 2 else 0

           if conf > 80 and class_detect == 'person':
                threshold = height - width
                if aspect_ratio > aspect_ratio_threshold or velocity > velocity_threshold or angle_change > angle_change_threshold or threshold < 0:
                    fall_text_position = (x1 + width // 2 - 60, y1 - 45)
                    cvzone.putTextRect(frame, 'Fall Detected', fall_text_position, scale=1.5, thickness=2, offset=10, colorR=(0, 0, 255))
                    prediction="Fall Detected"
                    print("Detction:" + prediction)
                    # Send SMS using Twilio
                    #try:
                       # message = client.messages.create(
                          #  body="Fall Detected! Immediate action may be required.",
                         #   from_=twilio_number,
                         #   to=your_number
                       # )
                        #resp = requests.post("http://localhost:5000/trigger-alert")
                        #print("Status update response:", resp.status_code)
                        #print(f"Message sent: {message.sid}")
                    #except Exception as e:
                    #     print(f"Failed to send SMS: {e}")




        _, jpeg = cv2.imencode('.jpg', frame)
        img_bytes = jpeg.tobytes()
        img_io = io.BytesIO(img_bytes)  # ✅ fix: wrap in BytesIO

    # Prepare response
        response = make_response(send_file(
        img_io,
        mimetype='image/jpeg',
        as_attachment=False,
        download_name="result.jpg"
    ))
        response.headers["X-Prediction"] = prediction  # 👈 send prediction
        return response


@app.route("/upload", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    if not file:
        return "No video uploaded", 400

    # Save uploaded video to temp file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    file.save(temp_input.name)

    cap = cv2.VideoCapture(temp_input.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare temp output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    fall_detected = False
    position_history = {}
    angle_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = classnames[cls]

                if label != "person" or conf < 0.6:
                    continue

                w = x2 - x1
                h = y2 - y1
                center_y = (y1 + y2) // 2
                aspect_ratio = w / h
                angle = math.degrees(math.atan2(h, w))

                pid = f"{x1}_{y1}_{x2}_{y2}"
                position_history.setdefault(pid, []).append(center_y)
                angle_history.setdefault(pid, []).append(angle)

                if len(position_history[pid]) > 2:
                    position_history[pid] = position_history[pid][-2:]
                if len(angle_history[pid]) > 2:
                    angle_history[pid] = angle_history[pid][-2:]

                velocity = position_history[pid][-1] - position_history[pid][-2] if len(position_history[pid]) >= 2 else 0
                angle_change = abs(angle_history[pid][-1] - angle_history[pid][-2]) if len(angle_history[pid]) >= 2 else 0

                threshold = h - w
                fall = (
                    aspect_ratio > 1.5
                    or velocity > 20
                    or angle_change > 45
                    or threshold < 0
                )

                if fall:
                    fall_detected = True
                    cv2.putText(frame, "Fall Detected", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                else:
                    cv2.putText(frame, "Normal", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    prediction = "Fall Detected" if fall_detected else "No fall detected"

    with open(temp_output.name, "rb") as f:
        video_data = f.read()

    response = make_response(video_data)
    response.headers["Content-Type"] = "video/mp4"
    response.headers["X-Prediction"] = prediction

    os.unlink(temp_input.name)
    os.unlink(temp_output.name)

    return response


if __name__ == "__main__":
    app.run(port=5000)
