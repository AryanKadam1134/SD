class StressRecommender:
    def __init__(self):
        self.recommendations = {
            'high': [
                "Take deep breaths for 5 minutes",
                "Step away from work for a short break",
                "Try progressive muscle relaxation",
                "Listen to calming music",
                "Practice mindfulness meditation"
            ],
            'medium': [
                "Take a short walk",
                "Stretch at your desk",
                "Drink some water",
                "Do quick shoulder rolls",
                "Practice 4-7-8 breathing"
            ],
            'low': [
                "Continue with regular breaks",
                "Maintain good posture",
                "Stay hydrated",
                "Take regular screen breaks",
                "Do gentle neck stretches"
            ]
        }
    
    def get_recommendation(self, stress_level):
        if stress_level >= 0.7:
            category = 'high'
        elif stress_level >= 0.4:
            category = 'medium'
        else:
            category = 'low'
            
        import random
        return random.choice(self.recommendations[category])