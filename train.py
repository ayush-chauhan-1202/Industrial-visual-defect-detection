import torch
from src.dataset import get_loaders
from src.model import UNetAE
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"

train_loader, _ = get_loaders("mvtec/tile")

model = UNetAE().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()

def ssim(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x*x, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y*y, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x * mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return (num / den).mean()

for epoch in range(75):
    model.train()
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        recon = model(x)
        l1 = loss_fn(recon, x)
        ssim_loss = 1 - ssim(recon, x)
        loss = l1 + 0.5 * ssim_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch}: {total / len(train_loader):.4f}")

torch.save(model.state_dict(), "model_tile75.pth")