# Create test_config.py manually with this content:

from config.config import config

print('🎯 Configuration loaded successfully!')
print(f'Video Frame Rate: {config.VIDEO.FRAME_RATE}')
print(f'Model Learning Rate: {config.MODEL.CNN_LEARNING_RATE}')
print(f'Event Keywords: {config.EVENT.KEYWORDS[:3]}...')

# Check if directories were created
import os
print(f'\n📁 Checking directories:')
for attr in ['RAW_VIDEO_DIR', 'HIGHLIGHTS_DIR', 'EXPERIMENTS_DIR']:
    dir_path = getattr(config.PATH, attr, None)
    if dir_path and os.path.exists(dir_path):
        print(f'✅ {attr}: {dir_path}')
    else:
        print(f'❌ {attr}: {dir_path}')