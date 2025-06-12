from flask import Flask, render_template, Response, jsonify, request, redirect, session, flash, url_for
import cv2
from app.detector import PalmVeinRecognizer
from app.database import *
from app.camera import capture_frame

app = Flask(__name__)  # ✅ FIXED: Correct Flask app init
app.secret_key = 'your_secret_key_here'

recognizer = PalmVeinRecognizer()
last_user = None
verified_matric = None  # Used to trigger redirect after verification


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video_feed():
    def gen_frames():
        global last_user, verified_matric
        while True:
            frame = capture_frame()
            if frame is None:
                continue

            user, confidence = recognizer.predict(frame)

            if confidence > 0.85:
                color = (0, 255, 0)
                status = f"{user} Verified"
                if verified_matric != user:  # avoid multiple triggers
                    student_info = get_student_info_by_matric(user)
                    if student_info:
                        last_user = student_info
                        verified_matric = user
                        log_activity(student_info['name'], student_info['matric'], log_type="verification")
            else:
                color = (0, 0, 255)
                status = "Adjust Hand..."

            cv2.rectangle(frame, (10, 10), (320, 80), color, 2)
            cv2.putText(frame, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_verified')
def check_verified():
    global verified_matric
    if verified_matric:
        matric = verified_matric
        verified_matric = None  # Reset once read
        return jsonify({"verified": True, "matric": matric})
    return jsonify({"verified": False})


@app.route('/verified/<matric>')
def verified(matric):
    student = get_student_info_by_matric(matric)
    if student:
        return render_template('verified.html', student=student)
    return "Student info not found", 404


@app.route('/info')
def info():
    return jsonify(last_user or {})


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form.to_dict()
        register_student(data)
        log_activity(data['name'], data['matric'], log_type="enrollment")

        # Save palm image
        frame = capture_frame()
        save_path = f"dataset/{data['matric']}.jpg"
        if frame is not None:
            cv2.imwrite(save_path, frame)

        return redirect('/')
    return render_template('register.html')


@app.route('/capture_frame')
def capture_frame_api():
    frame = capture_frame()
    if frame is not None:
        _, jpeg = cv2.imencode('.jpg', frame)
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    return "Camera error", 500


@app.route('/attendance')
def attendance():
    return render_template('attendance.html')


if __name__ == '__main__':  # ✅ FIXED
    init_db()
    app.run(host='0.0.0.0', port=5000)
