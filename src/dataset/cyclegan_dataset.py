import os
import random

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CycleGANDataset(Dataset):

    def __init__(self, young_dir, old_dir):
        self.young_dir = young_dir
        self.old_dir = old_dir

        self.young_images = sorted(os.listdir(young_dir))
        self.old_images = sorted(os.listdir(old_dir))
        # used sorted cause os.listdir() doesnt guarantee ordering

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return max(
            len(self.young_images),
            len(self.old_images)
        )

    def __getitem__(self, idx):
        young_path = os.path.join(
            self.young_dir,
            self.young_images[idx % len(self.young_images)]
        ) # modulo to be safe if we go beyond young_images' len

        old_path = os.path.join(
            self.old_dir,
            random.choice(self.old_images)
        ) # old images are less to random choicing them

        young_image = Image.open(
            young_path
        ).convert("RGB") # three channels or RGB forced

        old_image = Image.open(
            old_path
        ).convert("RGB")

        young_image = self.transform(
            young_image
        ) # PIL -> Tensor -> Normalized Tensors

        old_image = self.transform(
            old_image
        )

        return young_image, old_image

