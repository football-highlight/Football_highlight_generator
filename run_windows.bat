@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: Colors for output
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

:: Print colored message
echo %BLUE%[INFO]%NC% Starting Football Highlights Generator Setup...
echo.

:: Check Python
echo %BLUE%[INFO]%NC% Checking Python installation...
python --version > nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Python is not installed or not in PATH
    echo Please install Python 3.9 or later from https://python.org
    pause
    exit /b 1
)
python -c "import sys; print('Python ' + sys.version)" 2>&1
echo %GREEN%[SUCCESS]%NC% Python found

:: Create virtual environment
echo.
echo %BLUE%[INFO]%NC% Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo %GREEN%[SUCCESS]%NC% Virtual environment created
) else (
    echo %YELLOW%[INFO]%NC% Virtual environment already exists
)

:: Activate virtual environment
echo %BLUE%[INFO]%NC% Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to activate virtual environment
    pause
    exit /b 1
)
echo %GREEN%[SUCCESS]%NC% Virtual environment activated

:: Install requirements
echo.
echo %BLUE%[INFO]%NC% Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Failed to install requirements
    pause
    exit /b 1
)
echo %GREEN%[SUCCESS]%NC% Requirements installed

:: Create directories
echo.
echo %BLUE%[INFO]%NC% Creating project directories...
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

echo %GREEN%[SUCCESS]%NC% Directories created

:: Create sample video if none exists
echo.
echo %BLUE%[INFO]%NC% Checking for sample data...
if not exist "data\raw_videos\sample_match.mp4" (
    echo %YELLOW%[INFO]%NC% Creating sample video...
    
    python -c "
import cv2
import numpy as np
import os

# Create a simple sample video
duration = 5  # 5 seconds for testing
fps = 30
width, height = 640, 480

# Create output directory
os.makedirs('data/raw_videos', exist_ok=True)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/raw_videos/sample_match.mp4', fourcc, fps, (width, height))

print('Creating sample video...')
for i in range(duration * fps):
    # Create frame with gradient
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient
    for y in range(height):
        color = int((y / height) * 255)
        frame[y, :, 0] = color  # Blue channel
        frame[y, :, 1] = 255 - color  # Green channel
    
    # Add moving rectangle
    rect_x = (i * 5) % width
    cv2.rectangle(frame, (rect_x, 50), (rect_x + 100, 150), (0, 255, 0), -1)
    
    # Add text
    cv2.putText(frame, 'Football Test Video', (50, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Frame: {i}', (50, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(frame)

out.release()
print(f'Sample video created: data/raw_videos/sample_match.mp4')
print(f'Duration: {duration}s, Resolution: {width}x{height}, FPS: {fps}')
"
    
    if exist "data\raw_videos\sample_match.mp4" (
        echo %GREEN%[SUCCESS]%NC% Sample video created
    ) else (
        echo %YELLOW%[WARNING]%NC% Could not create sample video (OpenCV might not be installed yet)
    )
) else (
    echo %YELLOW%[INFO]%NC% Sample video already exists
)

:: Create sample annotation
echo %BLUE%[INFO]%NC% Creating sample annotation...
python -c "
import json
import os

# Create sample annotation
annotation = {
    'video_path': 'data/raw_videos/sample_match.mp4',
    'duration': 5,
    'events': [
        {'event_type': 'goal', 'start_time': 1.5, 'end_time': 2.5, 'confidence': 0.9},
        {'event_type': 'foul', 'start_time': 3.2, 'end_time': 3.7, 'confidence': 0.8}
    ]
}

os.makedirs('data/annotations', exist_ok=True)
with open('data/annotations/sample_match.json', 'w') as f:
    json.dump(annotation, f, indent=2)

print('Sample annotation created: data/annotations/sample_match.json')
"

:: Create configuration files
echo.
echo %BLUE%[INFO]%NC% Setting up configuration...
if not exist ".env" (
    copy .env.example .env
    echo %YELLOW%[INFO]%NC% Please edit .env file with your configuration
)

if not exist "config\config.py" (
    mkdir config 2>nul
    mkdir config\model_configs 2>nul
    mkdir config\training_configs 2>nul
    
    echo Creating basic config files...
    
    python -c "
import os
os.makedirs('config', exist_ok=True)
os.makedirs('config/model_configs', exist_ok=True)
os.makedirs('config/training_configs', exist_ok=True)

# Create basic config.py
config_content = '''
# config/config.py - Basic configuration

from dataclasses import dataclass
from pathlib import Path

@dataclass
class VideoConfig:
    FRAME_RATE = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    CLIP_DURATION = 5
    BUFFER_BEFORE = 5
    BUFFER_AFTER = 10

@dataclass
class AudioConfig:
    SAMPLE_RATE = 16000
    LANGUAGE = \"en\"
    WHISPER_MODEL = \"base\"

@dataclass  
class ModelConfig:
    CNN_INPUT_SHAPE = (16, 224, 224, 3)
    CNN_BATCH_SIZE = 8
    CNN_LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50
    
@dataclass
class EventConfig:
    KEYWORDS = [\"goal\", \"red card\", \"yellow card\", \"foul\", \"penalty\", 
                \"free kick\", \"corner\", \"offside\", \"save\", \"miss\"]
    CONFIDENCE_THRESHOLD = 0.7

@dataclass
class PathConfig:
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_VIDEO_DIR = PROJECT_ROOT / \"data\" / \"raw_videos\"
    HIGHLIGHTS_DIR = PROJECT_ROOT / \"data\" / \"highlights\"
    ANNOTATIONS_DIR = PROJECT_ROOT / \"data\" / \"annotations\"
    MODEL_DIR = PROJECT_ROOT / \"models\"
    LOGS_DIR = PROJECT_ROOT / \"logs\"

@dataclass
class AppConfig:
    API_HOST = \"0.0.0.0\"
    API_PORT = 8000
    STREAMLIT_PORT = 8501

@dataclass
class FootballHighlightsConfig:
    VIDEO = VideoConfig()
    AUDIO = AudioConfig()
    MODEL = ModelConfig()
    EVENT = EventConfig()
    PATH = PathConfig()
    APP = AppConfig()

config = FootballHighlightsConfig()
'''

with open('config/config.py', 'w') as f:
    f.write(config_content)

# Create sample YAML config
yaml_content = '''# config/model_configs/multimodal_config.yaml
model:
  name: \"football_multimodal\"
  num_classes: 10
  learning_rate: 0.0001
'''

with open('config/model_configs/multimodal_config.yaml', 'w') as f:
    f.write(yaml_content)

print('Basic configuration files created')
"
)

:: Create dummy model files
echo %BLUE%[INFO]%NC% Creating dummy model files...
python -c "
import torch
import os

os.makedirs('models/pretrained', exist_ok=True)

# Create dummy 3D CNN model
dummy_3dcnn = torch.nn.Sequential(
    torch.nn.Conv3d(3, 16, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
    torch.nn.Flatten(),
    torch.nn.Linear(16, 10)
)
torch.save(dummy_3dcnn.state_dict(), 'models/pretrained/r3d_18.pth')
print('Dummy 3D CNN model created')

# Create simple test script
test_script = '''
import sys
sys.path.append('.')

print('Testing imports...')
try:
    import torch
    import cv2
    import numpy as np
    print('✅ Core imports successful')
    
    from config.config import config
    print(f'✅ Config loaded: Frame rate = {config.VIDEO.FRAME_RATE}')
    
    print('✅ All tests passed!')
except Exception as e:
    print(f'❌ Test failed: {e}')
'''

with open('test_imports.py', 'w') as f:
    f.write(test_script)
"

:: Run tests
echo.
echo %BLUE%[INFO]%NC% Running basic tests...
python test_imports.py
del test_imports.py 2>nul

:: Show menu
echo.
echo ==========================================
echo    FOOTBALL HIGHLIGHTS GENERATOR
echo ==========================================
echo.
echo Setup complete!
echo.
echo Available commands:
echo.
echo 1. Preprocess data:
echo    python scripts\preprocess_data.py --input_dir data\raw_videos
echo.
echo 2. Train model:
echo    python scripts\train_model.py --epochs 5 --batch_size 2
echo.
echo 3. Process video:
echo    python scripts\run_inference.py --video_path data\raw_videos\sample_match.mp4
echo.
echo 4. Start web interface (in separate terminals):
echo    Terminal 1: uvicorn app.main:app --reload
echo    Terminal 2: streamlit run app\streamlit_app.py
echo.
echo 5. Or use the interactive menu below:
echo.

:menu
echo ==========================================
echo    MAIN MENU
echo ==========================================
echo.
echo 1. Complete Setup (Run again)
echo 2. Process sample video
echo 3. Train model (5 epochs for testing)
echo 4. Start web interface
echo 5. Exit
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto process
if "%choice%"=="3" goto train
if "%choice%"=="4" goto web
if "%choice%"=="5" goto end

echo Invalid choice
goto menu

:setup
call %0
goto menu

:process
echo.
echo %BLUE%[INFO]%NC% Processing sample video...
python scripts\run_inference.py --video_path data\raw_videos\sample_match.mp4 --verbose
echo.
pause
goto menu

:train
echo.
echo %BLUE%[INFO]%NC% Training model (5 epochs)...
python scripts\train_model.py --epochs 5 --batch_size 2 --verbose
echo.
pause
goto menu

:web
echo.
echo %YELLOW%[INFO]%NC% Starting web interface...
echo Please open TWO separate terminals and run:
echo.
echo Terminal 1^> uvicorn app.main:app --reload
echo Terminal 2^> streamlit run app\streamlit_app.py
echo.
echo Then open browser to http://localhost:8501
echo.
pause
goto menu

:end
echo.
echo %GREEN%[SUCCESS]%NC% Setup complete! Remember to activate virtual environment:
echo call venv\Scripts\activate.bat
echo.
pause