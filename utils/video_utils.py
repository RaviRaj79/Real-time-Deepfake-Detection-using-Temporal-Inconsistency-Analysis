import numpy as np
import cv2

def extract_temporal_features(current_frame, prev_frame):
    """
    Extracts simple temporal features between two frames.
    Returns the mean absolute difference as a feature.
    """
    if prev_frame is None:
        return np.array([[0.0]], dtype=np.float32)
    diff = cv2.absdiff(current_frame, prev_frame)
    feature = np.array([[np.mean(diff)]], dtype=np.float32)
    return feature
