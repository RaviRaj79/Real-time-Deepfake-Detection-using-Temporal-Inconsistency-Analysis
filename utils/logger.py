import json
import os
from datetime import datetime

class ResultsLogger:
    """Log and track deepfake detection results."""
    
    def __init__(self, log_dir="results"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_result(self, video_file, label, predictions, metrics=None):
        """Log results for a video."""
        result = {
            'video': video_file,
            'label': label,
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'metrics': metrics or {}
        }
        self.results.append(result)
    
    def calculate_statistics(self):
        """Calculate detection statistics."""
        if not self.results:
            return {}
        
        all_predictions = []
        for result in self.results:
            all_predictions.extend(result['predictions'])
        
        stats = {
            'total_videos': len(self.results),
            'total_frames': len(all_predictions),
            'avg_deepfake_prob': float(np.mean(all_predictions)),
            'max_deepfake_prob': float(np.max(all_predictions)),
            'min_deepfake_prob': float(np.min(all_predictions)),
            'std_deepfake_prob': float(np.std(all_predictions))
        }
        return stats
    
    def save_results(self):
        """Save results to JSON file."""
        filename = os.path.join(self.log_dir, f"results_{self.timestamp}.json")
        data = {
            'timestamp': self.timestamp,
            'results': self.results,
            'statistics': self.calculate_statistics()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Results saved to {filename}")
        return filename

import numpy as np
