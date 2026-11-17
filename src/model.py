import torch.nn as nn
from torchvision.models import (mobilenet_v3_small, MobileNet_V3_Small_Weights)

class YogaMultiTaskModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feature_dim = 576

        # Head 1: keypoint coordinates
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 34),
            nn.Sigmoid()
        )

        # Head 2: yoga pose classification
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features).flatten(1)

        keypoints = self.pose_head(features)
        keypoints = keypoints.view(-1, 17, 2)

        classes = self.class_head(features)

        return {"keypoints": keypoints, "classes": classes}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YogaMultiTaskModel().to(device)
print("Device:", device)