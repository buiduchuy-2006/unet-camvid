import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CamVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.names = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = Image.open(os.path.join(self.img_dir, n)).convert("RGB")
        mask = np.array(Image.open(os.path.join(self.mask_dir, n)))
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask