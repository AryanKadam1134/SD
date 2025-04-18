# README.md

# Stress Detection & Recommendation System

This project is a web application designed to detect stress levels through facial expressions using the AffectNet dataset. The application aims to provide accurate stress detection within 5 seconds and offers recommendations to overcome stress if detected.

## Project Structure

```
stress-detection-app
├── src
│   ├── api
│   ├── models
│   ├── data
│   ├── utils
│   └── webapp
├── tests
├── config.py
├── requirements.txt
└── README.md
```

## Features

- **Stress Detection**: Utilizes facial expression analysis to determine stress levels.
- **Recommendations**: Provides suggestions to manage stress based on detection results.
- **User-Friendly Interface**: Simple web interface for uploading images and viewing results.

## Dataset

The project uses the AffectNet dataset, which includes three files: train, test, and validation. Each file contains images labeled with eight emotions: fear, anger, happy, sad, neutral, disgust, contempt, and surprised.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stress-detection-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the application settings in `config.py`.

## Usage

1. Run the web application:
   ```
   python -m src.webapp
   ```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Upload an image to detect stress and view the results.

## Testing

To run the unit tests for the stress detection functionality, execute:
```
pytest tests/test_detector.py
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.#   S D 
 
 #   S D 
 
 #   S D 
 
 


.\venv_310\Scripts\activate