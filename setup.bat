@echo off
echo ==========================================
echo   FOOTBALL HIGHLIGHTS GENERATOR - WINDOWS
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo [INFO] Checking Python version...
python -c "import sys; print('Python', sys.version)"

REM Create virtual environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo [INFO] Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

REM Create directories
echo [INFO] Creating project directories...
mkdir data 2>nul
mkdir data\raw_videos 2>nul
mkdir data\processed_videos 2>nul
mkdir data\annotations 2>nul
mkdir data\highlights 2>nul
mkdir data\audio 2>nul
mkdir data\features 2>nul
mkdir data\datasets 2>nul

mkdir models 2>nul
mkdir models\3dcnn 2>nul
mkdir models\audio_models 2>nul
mkdir models\multimodal 2>nul
mkdir models\pretrained 2>nul
mkdir models\deployment 2>nul

mkdir experiments 2>nul
mkdir logs 2>nul
mkdir notebooks 2>nul
mkdir tests 2>nul
mkdir app\static 2>nul

REM Copy environment file
if not exist ".env" (
    echo [INFO] Creating .env file...
    copy .env.example .env
    echo [WARNING] Please edit .env file with your configuration
)

REM Download sample data
echo [INFO] Creating sample data...
python -c "
import cv2
import numpy as np

# Create a sample video
duration = 10
fps = 30
width, height = 640, 480

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/raw_videos/sample_match.mp4', fourcc, fps, (width, height))

for i in range(duration * fps):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.rectangle(frame, (i %% width, 0), ((i + 100) %% width, 100), (0, 255, 0), -1)
    cv2.circle(frame, (width//2, height//2), 50 + i %% 100, (255, 0, 0), -1)
    
    # Add text
    cv2.putText(frame, f'Frame: {i}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out.write(frame)

out.release()
print('[INFO] Sample video created at data/raw_videos/sample_match.mp4')
"

REM Create sample annotation
echo [INFO] Creating sample annotation...
python -c "
import json
import os

annotation = {
    'video_path': 'data/raw_videos/sample_match.mp4',
    'events': [
        {'event_type': 'goal', 'start_time': 2.5, 'end_time': 7.5, 'confidence': 0.9},
        {'event_type': 'foul', 'start_time': 5.2, 'end_time': 10.2, 'confidence': 0.8},
        {'event_type': 'corner', 'start_time': 8.7, 'end_time': 13.7, 'confidence': 0.7}
    ]
}

os.makedirs('data/annotations', exist_ok=True)
with open('data/annotations/sample_match.json', 'w') as f:
    json.dump(annotation, f, indent=2)

print('[INFO] Sample annotation created')
"

REM Run tests
echo [INFO] Running basic tests...
python -c "
try:
    import torch
    import cv2
    import numpy as np
    
    print('[SUCCESS] Core imports successful')
    
    # Test basic functionality
    print('Testing configuration...')
    from config.config import config
    print(f'Frame rate: {config.VIDEO.FRAME_RATE}')
    
    print('Testing video processing...')
    from src.video_processing.preprocessor import VideoProcessor
    print('VideoProcessor import successful')
    
    print('Testing audio processing...')
    from src.audio_processing.audio_processor import AudioProcessor
    print('AudioProcessor import successful')
    
    print('[SUCCESS] All tests passed!')
    
except Exception as e:
    print(f'[ERROR] Test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo.
echo ==========================================
echo         SETUP COMPLETE!
echo ==========================================
echo.
echo Next steps:
echo 1. Edit .env file if needed
echo 2. Run: python scripts\preprocess_data.py
echo 3. Run: python scripts\train_model.py
echo 4. Run: streamlit run app\streamlit_app.py
echo.
pause