import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.dataset import get_loaders
from src.model import UNetAE
from src.anomaly import compute_anomaly_map, compute_image_score

device = "mps" if torch.backends.mps.is_available() else "cpu"

_, test_loader = get_loaders("mvtec/tile")

model = UNetAE().to(device)
model.load_state_dict(torch.load("model_tile30.pth", map_location=device))

scores, labels = [], []

for x, y in test_loader:
    x = x.to(device)
    amap = compute_anomaly_map(model, x)
    s = compute_image_score(amap)

    scores.extend(s.cpu().numpy())
    labels.extend(y)

print("AUROC:", roc_auc_score(labels, scores))