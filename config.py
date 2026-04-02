"""Configuration file for the deepfake detection project."""

# Model configuration
MODEL_CONFIG = {
    'input_size': 512,
    'hidden_size': 256,
    'num_heads': 4,
    'dropout': 0.3,
    'batch_norm': True
}

# Video processing configuration
VIDEO_CONFIG = {
    'frame_size': (320, 240),
    'fps': 30,
    'skip_frames': 0  # Process every N frames (0 = all frames)
}

# Detection thresholds
DETECTION_CONFIG = {
    'deepfake_threshold': 0.5,
    'confidence_threshold': 0.8
}

# Logging configuration
LOG_CONFIG = {
    'log_dir': 'results',
    'save_predictions': True,
    'save_metrics': True,
    'verbose': True
}

# Feature extraction configuration
FEATURE_CONFIG = {
    'extract_optical_flow': True,
    'extract_histogram': True,
    'extract_temporal_diff': True,
    'feature_vector_size': 512
}
