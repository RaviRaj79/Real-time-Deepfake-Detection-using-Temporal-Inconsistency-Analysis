import cv2
import numpy as np
import torch
import csv
import os
from utils import advanced_features
from utils.logger import ResultsLogger
from models.advanced_model import load_advanced_model

class DeepfakeDetector:
    """Advanced deepfake detection pipeline."""
    
    def __init__(self):
        self.model = load_advanced_model()
        self.logger = ResultsLogger()
        self.frame_buffer = []
        self.buffer_size = 5
    
    def analyze_video(self, video_path, label=None):
        """Analyze a video file for deepfake content."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video file: {video_path}")
            return None
        
        # Validate video
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print(f"[ERROR] File exists but is not a valid video: {video_path}")
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        prev_frame = None
        frame_idx = 0
        predictions = []
        metrics_list = []
        
        print(f"\n[ANALYSIS] Video: {video_path} {'['+label+']' if label else ''}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            gray = advanced_features.preprocess_frame(frame)
            
            # Calculate metrics
            metrics = advanced_features.calculate_frame_metrics(gray)
            metrics_list.append(metrics)
            
            # Compute optical flow
            flow = advanced_features.compute_optical_flow(prev_frame, gray)
            
            # Extract features
            features = advanced_features.extract_advanced_temporal_features(gray, prev_frame, flow)
            feature_tensor = torch.from_numpy(features).float()
            
            # Predict
            prob = self.model.predict(feature_tensor)
            predictions.append(float(prob))
            
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}: Deepfake probability: {prob:.4f} | "
                      f"Metrics - Brightness: {metrics['brightness']:.1f}, "
                      f"Contrast: {metrics['contrast']:.1f}, "
                      f"Sharpness: {metrics['sharpness']:.2f}")
            
            prev_frame = gray
            frame_idx += 1
        
        cap.release()
        
        # Log results
        avg_prediction = np.mean(predictions) if predictions else 0
        self.logger.add_result(video_path, label, predictions, 
                              {'avg_prediction': avg_prediction, 'metrics': metrics_list})
        
        return {
            'video': video_path,
            'frames_analyzed': frame_idx,
            'avg_deepfake_prob': avg_prediction,
            'predictions': predictions
        }

def check_image_file(image_path):
    """Validate image file."""
    if not os.path.isfile(image_path):
        print(f"[WARNING] Image file not found: {image_path}")
        return False
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] File exists but is not a valid image: {image_path}")
        return False
    return True

def process_dataset(csv_path):
    """Process entire dataset from CSV."""
    detector = DeepfakeDetector()
    
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Check image file
            image_file = row.get("image")
            if image_file:
                if not os.path.isfile(image_file):
                    image_file = os.path.join(".", image_file)
                check_image_file(image_file)
            
            # Analyze both deepfake and real videos
            for label, col in [("deepfake", "deepfake"), ("real", "video")]:
                video_file = row[col]
                if not os.path.isfile(video_file):
                    video_file = os.path.join(".", video_file)
                
                if os.path.isfile(video_file):
                    detector.analyze_video(video_file, label=label)
                else:
                    print(f"[ERROR] Video file not found: {video_file}")
    
    # Save results
    detector.logger.save_results()
    
    # Print statistics
    stats = detector.logger.calculate_statistics()
    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)
    print(f"Total videos analyzed: {stats.get('total_videos', 0)}")
    print(f"Total frames analyzed: {stats.get('total_frames', 0)}")
    print(f"Average deepfake probability: {stats.get('avg_deepfake_prob', 0):.4f}")
    print(f"Max deepfake probability: {stats.get('max_deepfake_prob', 0):.4f}")
    print(f"Min deepfake probability: {stats.get('min_deepfake_prob', 0):.4f}")
    print(f"Std deepfake probability: {stats.get('std_deepfake_prob', 0):.4f}")
    print("="*60)

if __name__ == "__main__":
    print("[INFO] Starting advanced deepfake detection pipeline...")
    csv_path = "DeepFake Videos Dataset.csv"
    
    if os.path.isfile(csv_path):
        process_dataset(csv_path)
        print("\n[SUCCESS] Analysis complete!")
    else:
        print(f"[ERROR] CSV file not found: {csv_path}")
