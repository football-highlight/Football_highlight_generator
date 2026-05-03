# debug_config.py
from config.config import config

print("Available config attributes:")
for attr in dir(config):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\nTrying to access:")
try:
    print(f"config.VIDEO: {config.VIDEO}")
except AttributeError:
    print("config.VIDEO not found")

try:
    print(f"config.video_config: {config.video_config}")
except AttributeError:
    print("config.video_config not found")