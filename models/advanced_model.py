import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowExtractor(nn.Module):
    """Extract optical flow features from consecutive frames."""
    def __init__(self):
        super(OpticalFlowExtractor, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, optical_flow):
        x = F.relu(self.conv1(optical_flow))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.mean(dim=[2, 3])

class TemporalConsistencyNet(nn.Module):
    """Neural network for deepfake detection based on temporal inconsistencies."""
    def __init__(self, input_size=512):
        super(TemporalConsistencyNet, self).__init__()
        
        self.feature_extraction = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.temporal_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.classification = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        features = self.feature_extraction(x)
        features = features.unsqueeze(1)
        attn_out, _ = self.temporal_attention(features, features, features)
        attn_out = attn_out.squeeze(1)
        logits = self.classification(attn_out)
        return logits
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs[:, 1].item()  # Deepfake probability

def load_advanced_model(model_path=None):
    """Load the advanced deepfake detection model."""
    model = TemporalConsistencyNet()
    if model_path and torch.cuda.is_available():
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print("[INFO] Could not load model weights, using initialized model")
    model.eval()
    return model
