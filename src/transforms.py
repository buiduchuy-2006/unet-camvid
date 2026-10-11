import numpy as np, torch
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

def make_transform(size=256):
    def tf(img, mask):
        img = img.resize((size, size), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((size, size), Image.NEAREST)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
        img = (img - MEAN) / STD
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return tf