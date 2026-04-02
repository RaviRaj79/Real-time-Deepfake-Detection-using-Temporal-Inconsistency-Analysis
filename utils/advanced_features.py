import cv2
import numpy as np
import torch

def compute_optical_flow(prev_frame, curr_frame):
    """Compute optical flow between two consecutive frames."""
    if prev_frame is None:
        return None
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def extract_advanced_temporal_features(curr_frame, prev_frame, flow=None):
    """
    Extract advanced temporal features from consecutive frames.
    Features include optical flow, frame differences, and temporal coherence.
    """
    features = []
    
    # Frame difference magnitude
    if prev_frame is not None:
        diff = cv2.absdiff(curr_frame, prev_frame)
        features.append(np.mean(diff))
        features.append(np.std(diff))
        features.append(np.max(diff))
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Current frame statistics
    features.append(np.mean(curr_frame))
    features.append(np.std(curr_frame))
    features.append(np.max(curr_frame))
    features.append(np.min(curr_frame))
    
    # Histogram features
    hist = cv2.calcHist([curr_frame], [0], None, [16], [0, 256])
    features.extend(hist.flatten()[:8])
    
    # Optical flow features if available
    if flow is not None:
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        features.append(np.mean(magnitude))
        features.append(np.std(magnitude))
        features.append(np.mean(angle))
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Pad to 512 features
    while len(features) < 512:
        features.append(0.0)
    
    return np.array(features[:512], dtype=np.float32)

def extract_temporal_features(current_frame, prev_frame):
    """Legacy wrapper for backward compatibility."""
    return extract_advanced_temporal_features(current_frame, prev_frame)

def preprocess_frame(frame, size=(320, 240)):
    """Preprocess frame for analysis."""
    frame = cv2.resize(frame, size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def calculate_frame_metrics(frame):
    """Calculate quality metrics for a frame."""
    metrics = {
        'brightness': np.mean(frame),
        'contrast': np.std(frame),
        'sharpness': cv2.Laplacian(frame, cv2.CV_64F).var()
    }
    return metrics
