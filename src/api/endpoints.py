from flask import Blueprint, request, jsonify
from src.models.stress_detector import StressDetector

api = Blueprint('api', __name__)
detector = StressDetector()

@api.route('/detect', methods=['POST'])
def detect_stress():
    data = request.json
    image = data.get('image')
    
    if not image:
        return jsonify({'error': 'No image provided'}), 400
    
    stress_level, emotion = detector.predict(image)
    return jsonify({'stress_level': stress_level, 'emotion': emotion})

@api.route('/recommend', methods=['GET'])
def recommend():
    return jsonify({
        'recommendation': 'Take deep breaths and try to relax. Consider taking a short break.'
    })