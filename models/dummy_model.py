import torch
import torch.nn as nn

class DummyDeepfakeDetector(nn.Module):
    def __init__(self):
        super(DummyDeepfakeDetector, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        # Dummy forward pass
        return torch.sigmoid(self.fc(x))

def load_dummy_model():
    model = DummyDeepfakeDetector()
    # In real use, load weights here
    return model
