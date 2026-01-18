import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.model import UNetAE
from src.anomaly import compute_anomaly_map

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = UNetAE().to(device)
model.load_state_dict(torch.load("model_tile30.pth", map_location=device))
model.eval()

img = Image.open("mvtec/tile/test/crack/005.png").convert("RGB")
t = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

x = t(img).unsqueeze(0).to(device)
amap = compute_anomaly_map(model, x)[0,0].cpu()

plt.subplot(1,2,1)
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(amap, cmap="hot")
plt.axis("off")

plt.savefig("result.png")