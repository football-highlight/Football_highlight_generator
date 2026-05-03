# fix_missing_attributes.py
import os

# Files that might have missing config attributes
files_to_check = {
    'src/event_detection/event_detector.py': [
        ("config.EVENT.EVENT_WEIGHTS", 
         "getattr(config.EVENT, 'EVENT_WEIGHTS', {\n        'goal': 1.0,\n        'red_card': 0.9,\n        'yellow_card': 0.8,\n        'penalty': 0.95,\n        'free_kick': 0.7,\n        'corner': 0.6,\n        'substitution': 0.5,\n        'save': 0.85,\n        'foul': 0.7,\n        'offside': 0.6\n    })")
    ]
}

for filepath, replacements in files_to_check.items():
    if os.path.exists(filepath):
        print(f"Checking {filepath}...")
        with open(filepath, 'r') as f:
            content = f.read()
        
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                print(f"  Fixed: {old}")
        
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Updated {filepath}")
    else:
        print(f"✗ File not found: {filepath}")

print("\n✅ Fix applied!")