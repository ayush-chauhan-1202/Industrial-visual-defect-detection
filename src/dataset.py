import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root, split="train"):
        self.samples = []
        self.labels = []
        self.split = split

        img_root = os.path.join(root, split)

        for defect in os.listdir(img_root):
            folder = os.path.join(img_root, defect)
            for f in os.listdir(folder):
                self.samples.append(os.path.join(folder, f))
                self.labels.append(0 if defect == "good" else 1)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def get_loaders(root, batch_size=16):
    train = MVTecDataset(root, "train")
    test = MVTecDataset(root, "test")

    return (
        DataLoader(train, batch_size, shuffle=True),
        DataLoader(test, batch_size, shuffle=False)
    )