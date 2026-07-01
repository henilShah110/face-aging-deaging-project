import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5], [0.5])  # [-1,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        # Extract age from filename
        age = int(img_name.split("_")[1].split(".")[0])

        image = self.transform(image)

        return image, age