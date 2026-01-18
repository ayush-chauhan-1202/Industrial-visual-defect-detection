import torch

def compute_anomaly_map(model, x):
    model.eval()
    with torch.no_grad():
        recon = model(x)
        error = torch.abs(x - recon)
        return error.mean(dim=1, keepdim=True)

def compute_image_score(anomaly_map):
    return anomaly_map.view(anomaly_map.size(0), -1).mean(dim=1)