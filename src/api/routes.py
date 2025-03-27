from flask import Flask, render_template, Response, jsonify
import cv2
from src.utils.face_detection import FaceDetector
from src.models.stress_detector import StressDetector
from src.utils.recommendations import StressRecommender
import time

app = Flask(__name__, 
           template_folder='../webapp/templates',
           static_folder='../webapp/static')

detector = FaceDetector()
stress_detector = StressDetector(model_path='models/best_model.pth')
recommender = StressRecommender()

def gen_frames():
    camera = cv2.VideoCapture(0)
    start_time = time.time()
    stress_readings = []
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        face = detector.detect_face(frame)
        if face is not None:
            stress_level, emotion, confidence = stress_detector.predict_stress(face)
            stress_readings.append(stress_level)
            
            # Calculate average stress over 5 seconds
            if time.time() - start_time >= 5:
                avg_stress = sum(stress_readings) / len(stress_readings)
                recommendation = recommender.get_recommendation(avg_stress)
                
                # Draw results on frame
                cv2.putText(face, f"Stress: {avg_stress:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(face, emotion, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Reset readings
                start_time = time.time()
                stress_readings = []
            
            ret, buffer = cv2.imencode('.jpg', face)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stress_level')
def get_stress_level():
    return jsonify({
        'stress_level': stress_detector.current_stress_level,
        'recommendation': recommender.get_recommendation(stress_detector.current_stress_level)
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)