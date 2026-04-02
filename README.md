# Real-time Deepfake Detection using Temporal Inconsistency Analysis

Advanced deepfake detection system that analyzes temporal inconsistencies across video frames using deep learning and computer vision techniques.

## Key Features

- **Advanced Neural Network**: Multi-head attention-based architecture (TemporalConsistencyNet)
- **Optical Flow Analysis**: Compute and analyze motion patterns between frames
- **Multi-Scale Feature Extraction**: Histogram, temporal differences, and frame metrics
- **Real-time Processing**: Frame-by-frame analysis with progress reporting
- **Results Logging**: Comprehensive JSON results with statistics
- **Frame Quality Metrics**: Brightness, contrast, and sharpness analysis
- **Batch Processing**: Process entire datasets from CSV configuration

## Project Structure

```
.
├── main.py                           # Advanced main pipeline
├── config.py                         # Configuration settings
├── DeepFake Videos Dataset.csv       # Dataset configuration
├── requirements.txt                  # Python dependencies
├── run_project.ps1                   # PowerShell execution script
├── setup_test_data.py                # Test data generator
├── models/
│   ├── __init__.py
│   ├── advanced_model.py            # TemporalConsistencyNet & OpticalFlowExtractor
│   ├── dummy_model.py               # Legacy model
│   └── [model weights]              # Pre-trained weights (optional)
├── utils/
│   ├── __init__.py
│   ├── advanced_features.py         # Advanced feature extraction
│   ├── video_utils.py               # Legacy utilities
│   └── logger.py                    # Results logging
├── deepfake/                        # Deepfake video files
├── video/                           # Real video files
├── image/                           # Image files
└── results/                         # Analysis results (generated)
```

## Setup Instructions

### 1. Environment Setup
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
- **Windows (PowerShell)**:
  ```bash
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```bash
  .\venv\Scripts\activate.bat
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
Ensure your DeepFake Videos Dataset.csv references all image and video files in the correct folders.

## Running the Project

### Quick Start (Windows)
```powershell
.\run_project.ps1
```

### Manual Execution
```bash
python main.py
```

### Generate Test Data
```bash
python setup_test_data.py
```

## Advanced Features

### 1. Optical Flow Computation
Analyzes motion patterns between consecutive frames to detect inconsistencies characteristic of deepfakes.

### 2. Multi-Head Attention
Uses transformer-style attention mechanisms to model temporal dependencies across frame sequences.

### 3. Feature Extraction
Extracts 512-dimensional feature vectors combining:
- Temporal frame differences (3 features)
- Frame statistics - brightness, contrast, min/max (4 features)
- Histogram features (8 features)
- Optical flow magnitude and angle (3 features)
- Padding for fixed-size tensor (494 features)

### 4. Results Logging
Automatically saves analysis results to `results/` directory with:
- Per-frame predictions
- Frame quality metrics
- Overall statistics
- Timestamp information

## Architecture

### TemporalConsistencyNet
- Input: 512-dimensional feature vector
- Feature extraction: 3 fully-connected layers with batch normalization
- Temporal attention: 4-head multi-head attention
- Classification: 2-class output (real/deepfake)

### OpticalFlowExtractor
- Input: 2-channel optical flow (x, y components)
- 2 convolutional layers with max pooling
- Output: 128-dimensional feature vector

## Configuration

Edit `config.py` to customize:
- Model architecture parameters
- Video processing settings
- Detection thresholds
- Feature extraction options
- Logging preferences

## Output

Results are saved to `results/results_YYYYMMDD_HHMMSS.json` containing:
- Per-video analysis results
- Frame-by-frame predictions
- Aggregate statistics
- Frame metrics

### Sample Output
```
==============================================================
DETECTION STATISTICS
==============================================================
Total videos analyzed: 10
Total frames analyzed: 600
Average deepfake probability: 0.4523
Max deepfake probability: 0.9876
Min deepfake probability: 0.0012
Std deepfake probability: 0.3456
==============================================================
```

## Requirements

- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- PyTorch
- scikit-learn

## Usage Examples

### Analyzing a Single Video
```python
from main import DeepfakeDetector

detector = DeepfakeDetector()
result = detector.analyze_video('path/to/video.mp4', label='test')
print(result)
```

### Processing Entire Dataset
```bash
python main.py
```

### Accessing Results
Results are logged to JSON files in the `results/` directory. Open any `.json` file to view detailed predictions and statistics.

## Advanced Customization

### Using Custom Models
1. Train your model following the TemporalConsistencyNet architecture
2. Save weights to `models/weights.pth`
3. Update `load_advanced_model()` in `advanced_model.py` with your weights path

### Custom Feature Extraction
Modify `extract_advanced_temporal_features()` in `utils/advanced_features.py` to implement custom feature sets.

### Threshold Adjustment
Edit detection thresholds in `config.py`:
```python
DETECTION_CONFIG = {
    'deepfake_threshold': 0.5,  # Adjust as needed
    'confidence_threshold': 0.8
}
```

## Performance Notes

- Frame processing speed depends on video resolution and system specifications
- GPU acceleration available with CUDA-compatible PyTorch
- Results saved in real-time as videos are processed
- Memory usage scales with buffer size and feature dimensions

## Troubleshooting

**Issue**: "Cannot open video file" error
- Ensure video files are in supported formats (MP4, MOV, AVI)
- Check file paths in CSV match actual file locations
- Verify video file integrity

**Issue**: Low detection accuracy
- Ensure test data has sufficient variation
- Try adjusting feature extraction parameters
- Consider training with actual deepfake datasets
- Increase buffer size for more temporal context

## Notes

- This is a research/proof-of-concept implementation
- For production use, train on labeled deepfake datasets (e.g., FaceForensics++)
- Model weights are randomly initialized; real detection requires proper training
- Frame analysis is sequential; parallel processing can be added for speed
- Results should be validated against ground truth labels
- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- scikit-learn

## Notes
- Ensure all files referenced in the CSV exist and are valid.
- The code will only process files listed in the dataset CSV.
