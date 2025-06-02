from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

yellow_lower = np.array([20, 100, 100], dtype="uint8")
yellow_upper = np.array([35, 255, 255], dtype="uint8")


pink_lower = np.array([140, 170, 150], dtype="uint8")
pink_upper = np.array([170, 255, 255], dtype="uint8")


camera = cv2.VideoCapture(1)  # Change to 1 if 0 doesn't work


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, pink_lower, pink_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if 200 < area < 10000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Streaming response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
