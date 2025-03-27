def plot_stress_detection_results(results):
    import matplotlib.pyplot as plt

    emotions = list(results.keys())
    values = list(results.values())

    plt.bar(emotions, values, color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Detection Confidence')
    plt.title('Stress Detection Results')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def display_emotion_distribution(emotion_counts):
    import matplotlib.pyplot as plt

    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Emotion Distribution')
    plt.show()