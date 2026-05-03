# scripts/balance_dataset.py
import pickle
import numpy as np
from pathlib import Path

# Load datasets
data_dir = Path("data/datasets")
datasets = {}

for split in ['train', 'val', 'test']:
    with open(data_dir / f"{split}_dataset.pkl", 'rb') as f:
        datasets[split] = pickle.load(f)

# Analyze class distribution
print("Original class distribution:")
for split, data in datasets.items():
    samples = data['samples']
    label_map = data['label_map']
    
    # Count samples per class
    class_counts = {}
    for sample in samples:
        event_type = sample['event_type']
        class_counts[event_type] = class_counts.get(event_type, 0) + 1
    
    print(f"\n{split.upper()}:")
    for event_type, count in class_counts.items():
        print(f"  {event_type}: {count} samples")

# Strategy: Oversample minority classes
print("\n\nCreating balanced dataset...")

# Find max samples per class
max_samples_per_class = 200  # Target number

balanced_datasets = {}
for split, data in datasets.items():
    samples = data['samples']
    
    # Group samples by class
    samples_by_class = {}
    for sample in samples:
        event_type = sample['event_type']
        if event_type not in samples_by_class:
            samples_by_class[event_type] = []
        samples_by_class[event_type].append(sample)
    
    # Balance each class
    balanced_samples = []
    for event_type, class_samples in samples_by_class.items():
        num_samples = len(class_samples)
        
        if num_samples < max_samples_per_class:
            # Oversample
            repeat_times = max_samples_per_class // num_samples
            remainder = max_samples_per_class % num_samples
            
            # Repeat samples
            oversampled = class_samples * repeat_times
            oversampled += class_samples[:remainder]
            balanced_samples.extend(oversampled)
        else:
            # Use all samples (or undersample)
            balanced_samples.extend(class_samples[:max_samples_per_class])
    
    # Shuffle
    np.random.shuffle(balanced_samples)
    
    balanced_datasets[split] = {
        'samples': balanced_samples,
        'label_map': data['label_map'],
        'split': split,
        'num_samples': len(balanced_samples)
    }
    
    print(f"{split.upper()}: {len(balanced_samples)} balanced samples")

# Save balanced datasets
for split, data in balanced_datasets.items():
    with open(data_dir / f"{split}_dataset_balanced.pkl", 'wb') as f:
        pickle.dump(data, f)

print("\n✅ Balanced datasets saved!")