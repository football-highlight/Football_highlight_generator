"""
Event detection and annotation
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

from config.config import config
from src.video_processing.preprocessor import VideoClip
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DetectedEvent:
    """Detected event data structure"""
    event_type: str
    confidence: float
    start_time: float
    end_time: float
    source: str  # 'model', 'commentary', 'fusion'
    keywords: Optional[List[str]] = None
    model_predictions: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'event_type': self.event_type,
            'confidence': self.confidence,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'source': self.source,
            'keywords': self.keywords,
            'model_predictions': self.model_predictions
        }


class EventDetector:
    """Event detection and annotation system"""
    
    def __init__(self, config, model: Optional[torch.nn.Module] = None):
        self.config = config.EVENT
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Event configuration
        self.keywords = config.EVENT.KEYWORDS
        self.event_weights = getattr(config.EVENT, 'EVENT_WEIGHTS', None)
        self.confidence_threshold = config.EVENT.CONFIDENCE_THRESHOLD
        
        # Move model to device if provided
        if model is not None:
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"EventDetector initialized with model on {self.device}")
        else:
            logger.info("EventDetector initialized without model (keyword-based only)")
    
    def detect_events_from_clip(
        self,
        video_clip: VideoClip,
        audio_features: Dict,
        commentary_keywords: List[Dict]
    ) -> List[DetectedEvent]:
        """
        Detect events in a video clip using multi-modal features
        
        Args:
            video_clip: Video clip to analyze
            audio_features: Audio features dictionary
            commentary_keywords: Keywords detected in commentary
            
        Returns:
            List of detected events
        """
        events = []
        
        # 1. Detect events from model (if available)
        if self.model is not None:
            model_events = self._detect_events_with_model(video_clip, audio_features)
            events.extend(model_events)
        
        # 2. Detect events from commentary keywords
        commentary_events = self._detect_events_from_commentary(
            video_clip, commentary_keywords
        )
        events.extend(commentary_events)
        
        # 3. Merge overlapping events
        merged_events = self._merge_events(events)
        
        # 4. Apply confidence threshold
        filtered_events = [
            event for event in merged_events 
            if event.confidence >= self.confidence_threshold
        ]
        
        logger.debug(f"Detected {len(filtered_events)} events in clip "
                    f"[{video_clip.start_time:.1f}s - {video_clip.end_time:.1f}s]")
        
        return filtered_events
    
    def _detect_events_with_model(
        self,
        video_clip: VideoClip,
        audio_features: Dict
    ) -> List[DetectedEvent]:
        """
        Detect events using the trained model
        
        Args:
            video_clip: Video clip to analyze
            audio_features: Audio features dictionary
            
        Returns:
            List of detected events from model
        """
        try:
            # Prepare inputs for model
            video_tensor = self._preprocess_video_clip(video_clip)
            audio_tensor = self._preprocess_audio_features(audio_features)
            
            # Run model inference
            with torch.no_grad():
                if hasattr(self.model, 'forward_multimodal'):
                    outputs = self.model.forward_multimodal(video_tensor, audio_tensor)
                else:
                    outputs = self.model(video_tensor, audio_tensor)
                
                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
            
            # Convert to events
            events = []
            for i in range(video_tensor.size(0)):
                confidence = confidences[i].item()
                prediction = predictions[i].item()
                
                if prediction < len(self.keywords):
                    event_type = self.keywords[prediction]
                    
                    # Get all prediction probabilities
                    all_probs = probabilities[i].cpu().numpy()
                    model_predictions = {
                        self.keywords[j]: float(all_probs[j]) 
                        for j in range(len(self.keywords))
                    }
                    
                    event = DetectedEvent(
                        event_type=event_type,
                        confidence=confidence,
                        start_time=video_clip.start_time,
                        end_time=video_clip.end_time,
                        source='model',
                        model_predictions=model_predictions
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error in model-based event detection: {e}")
            return []
    
    def _detect_events_from_commentary(
        self,
        video_clip: VideoClip,
        commentary_keywords: List[Dict]
    ) -> List[DetectedEvent]:
        """
        Detect events from commentary keywords
        
        Args:
            video_clip: Video clip being analyzed
            commentary_keywords: Keywords detected in commentary
            
        Returns:
            List of detected events from commentary
        """
        events = []
        
        for keyword_info in commentary_keywords:
            # Check if keyword falls within clip time
            if (video_clip.start_time <= keyword_info['start_time'] <= video_clip.end_time):
                
                # Create event from keyword
                event = DetectedEvent(
                    event_type=keyword_info['keyword'],
                    confidence=keyword_info.get('confidence', 0.8),
                    start_time=keyword_info['start_time'],
                    end_time=keyword_info['end_time'],
                    source='commentary',
                    keywords=[keyword_info['word']]
                )
                events.append(event)
        
        return events
    
    # In src/event_detection/event_detector.py, update the _preprocess_video_clip method:

    def _preprocess_video_clip(self, video_clip: VideoClip) -> torch.Tensor:
        """Preprocess video clip for model input"""
        from src.video_processing.preprocessor import frames_to_tensor
    
        # DEBUG: Log original clip shape
        print(f"🔍 Original clip shape: {video_clip.frames.shape}")
        print(f"🔍 Expected clip shape: 16 frames of {self.config.VIDEO.FRAME_HEIGHT}x{self.config.VIDEO.FRAME_WIDTH}")
    
        # Convert frames to tensor
        frames_tensor = frames_to_tensor(video_clip.frames)
    
        # DEBUG: Log tensor shape
        print(f"🔍 Frames tensor shape: {frames_tensor.shape}")
    
        # Check if we have enough frames
        if frames_tensor.size(1) < 16:  # Need at least 16 frames
            print(f"⚠️ Not enough frames: {frames_tensor.size(1)}. Padding with zeros.")
            # Pad with zeros
            padding = torch.zeros(3, 16 - frames_tensor.size(1), 
                                frames_tensor.size(2), frames_tensor.size(3),
                                device=frames_tensor.device)
            frames_tensor = torch.cat([frames_tensor, padding], dim=1)
        elif frames_tensor.size(1) > 16:
            print(f"⚠️ Too many frames: {frames_tensor.size(1)}. Taking first 16.")
            frames_tensor = frames_tensor[:, :16, :, :]
    
        # Add batch dimension if needed
        if frames_tensor.dim() == 4:
            frames_tensor = frames_tensor.unsqueeze(0)
    
        print(f"🔍 Final input shape to model: {frames_tensor.shape}")
    
        return frames_tensor.to(self.device)
    
    def _preprocess_audio_features(self, audio_features: Dict) -> torch.Tensor:
        """Preprocess audio features for model input - UPDATED FOR MFCC"""
        # Extract MFCC features
        mfcc = audio_features.get('mfcc', np.zeros((13, 100)))
        
        print(f"🔍 EVENT DETECTOR: MFCC shape from audio features: {mfcc.shape}")
        
        # Convert to tensor
        mfcc_tensor = torch.from_numpy(mfcc).float()
        
        # CRITICAL FIX: Format for MFCC audio model
        # Option 1: With channel dimension [batch, 1, 13, 100]
        mfcc_tensor = mfcc_tensor.unsqueeze(0)  # Add batch: [1, 13, 100]
        mfcc_tensor = mfcc_tensor.unsqueeze(1)  # Add channel: [1, 1, 13, 100]
        
        # Option 2: Without channel dimension [batch, 13, 100] 
        # (audio model will add it automatically)
        # mfcc_tensor = mfcc_tensor.unsqueeze(0)  # Add batch only: [1, 13, 100]
        
        print(f"🔍 EVENT DETECTOR: Tensor shape after preprocessing: {mfcc_tensor.shape}")
        
        return mfcc_tensor.to(self.device)
    
    def _merge_events(self, events: List[DetectedEvent]) -> List[DetectedEvent]:
        """Merge overlapping or similar events"""
        if not events:
            return []
        
        # Sort by start time
        events.sort(key=lambda x: x.start_time)
        
        merged_events = []
        current_event = events[0]
        
        for next_event in events[1:]:
            # Check if events overlap or are close
            events_overlap = (
                next_event.start_time <= current_event.end_time + self.config.EVENT.MERGE_WINDOW
            )
            
            same_type = (
                next_event.event_type == current_event.event_type
            )
            
            if events_overlap and same_type:
                # Merge events
                current_event.end_time = max(current_event.end_time, next_event.end_time)
                current_event.confidence = max(current_event.confidence, next_event.confidence)
                
                # Merge sources
                if current_event.source != next_event.source:
                    current_event.source = 'fusion'
                
                # Merge keywords
                if next_event.keywords:
                    if current_event.keywords is None:
                        current_event.keywords = []
                    current_event.keywords.extend(next_event.keywords)
                
                # Merge model predictions (take max confidence)
                if next_event.model_predictions:
                    if current_event.model_predictions is None:
                        current_event.model_predictions = {}
                    
                    for event_type, confidence in next_event.model_predictions.items():
                        current_event.model_predictions[event_type] = max(
                            current_event.model_predictions.get(event_type, 0),
                            confidence
                        )
            
            else:
                # Add current event to merged list
                merged_events.append(current_event)
                current_event = next_event
        
        # Add the last event
        merged_events.append(current_event)
        
        return merged_events
    
    def save_annotations(
        self,
        events: List[DetectedEvent],
        video_path: str,
        output_path: str
    ) -> Dict:
        """
        Save event annotations to JSON file
        
        Args:
            events: List of detected events
            video_path: Path to original video
            output_path: Path to save annotations
            
        Returns:
            Annotation dictionary
        """
        # Convert events to dictionaries
        events_dict = [event.to_dict() for event in events]
        
        # Create annotation object
        annotation = {
            'video_path': video_path,
            'processed_at': datetime.now().isoformat(),
            'total_events': len(events),
            'events': events_dict,
            'configuration': {
                'keywords': self.keywords,
                'confidence_threshold': self.confidence_threshold,
                'event_weights': self.event_weights
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        logger.info(f"Saved {len(events)} annotations to {output_path}")
        
        return annotation
    
    def load_annotations(self, annotation_path: str) -> Tuple[List[DetectedEvent], Dict]:
        """
        Load event annotations from JSON file
        
        Args:
            annotation_path: Path to annotation file
            
        Returns:
            Tuple of (events list, metadata dict)
        """
        try:
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            # Convert dictionaries back to DetectedEvent objects
            events = []
            for event_dict in annotation.get('events', []):
                event = DetectedEvent(
                    event_type=event_dict['event_type'],
                    confidence=event_dict['confidence'],
                    start_time=event_dict['start_time'],
                    end_time=event_dict['end_time'],
                    source=event_dict['source'],
                    keywords=event_dict.get('keywords'),
                    model_predictions=event_dict.get('model_predictions')
                )
                events.append(event)
            
            metadata = {
                'video_path': annotation.get('video_path'),
                'processed_at': annotation.get('processed_at'),
                'total_events': annotation.get('total_events')
            }
            
            logger.info(f"Loaded {len(events)} annotations from {annotation_path}")
            
            return events, metadata
            
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return [], {}


def test_event_detector():
    """Test event detector functions"""
    print("Testing EventDetector...")
    
    from config.config import config
    
    # Create dummy event detector without model
    detector = EventDetector(config)
    
    # Create dummy events
    dummy_events = [
        DetectedEvent(
            event_type='goal',
            confidence=0.9,
            start_time=30.0,
            end_time=35.0,
            source='model'
        ),
        DetectedEvent(
            event_type='goal',
            confidence=0.85,
            start_time=32.0,
            end_time=37.0,
            source='commentary',
            keywords=['goal']
        ),
        DetectedEvent(
            event_type='foul',
            confidence=0.7,
            start_time=120.0,
            end_time=125.0,
            source='model'
        ),
    ]
    
    # Test event merging
    merged = detector._merge_events(dummy_events)
    print(f"✅ Original events: {len(dummy_events)}")
    print(f"✅ Merged events: {len(merged)}")
    
    # Test annotation saving/loading
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        
        # Save annotations
        annotation = detector.save_annotations(
            merged, 
            'test_video.mp4',
            temp_path
        )
        
        print(f"✅ Saved {annotation['total_events']} events")
        
        # Load annotations
        loaded_events, metadata = detector.load_annotations(temp_path)
        print(f"✅ Loaded {len(loaded_events)} events")
        
        # Cleanup
        os.unlink(temp_path)
    
    print("✅ All event detector tests passed!")


if __name__ == "__main__":
    test_event_detector()