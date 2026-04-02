Real-time Deepfake Detection using Temporal Inconsistency Analysis

## Overview
An advanced deep learning system for detecting deepfake videos by analyzing temporal inconsistencies across video frames. Uses multi-head attention networks and optical flow analysis to identify subtle artifacts characteristic of deepfakes.

## Key Features
- Advanced TemporalConsistencyNet architecture with multi-head attention
- Optical flow computation for motion pattern analysis
- Multi-scale feature extraction (512-dimensional vectors)
- Real-time frame-by-frame processing with progress reporting
- Comprehensive results logging and statistics
- Frame quality metrics (brightness, contrast, sharpness analysis)
- Batch processing from CSV dataset configuration

## Technology Stack
- Python 3.8+
- PyTorch (Deep Learning)
- OpenCV (Computer Vision)
- NumPy (Numerical Computing)

## Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python main.py`

## Project Structure
- `main.py` - Advanced detection pipeline
- `models/` - Neural network architectures
- `utils/` - Feature extraction and logging utilities
- `config.py` - Configuration settings
- `DeepFake Videos Dataset.csv` - Dataset configuration

## License
MIT License - See LICENSE file for details

## Author
Ravi Raj

## Status
Production-ready with test data and examples
