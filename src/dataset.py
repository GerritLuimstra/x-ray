import glob
import random
from typing import List

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as transforms
from pathlib import Path
import glob

POS_PATH = 'PNEUMONIA'
NEG_PATH = 'NORMAL'

def read_images_from_folder(folder: str) -> List[str]:
    images = []

    # Read in images from directory
    for filename in glob.glob(f'{folder}/*.jpeg'):
        img = Image.open(filename)
        img = img.resize((312, 312))
        images.append(img.copy())
        img.close()

    return images


class XRayDataset(Dataset):
    def __init__(self, img_dir, transform=None) -> None:
        X_pos = read_images_from_folder(img_dir + '/' + POS_PATH)
        X_neg = read_images_from_folder(img_dir + '/' + NEG_PATH)

        X = X_pos + X_neg
        y = list([1] * len(X_pos)) + list([0] * len(X_neg))
        assert len(X) == len(y)
        assert len(X) > 0

        self.X, self.y = list(X), list(y)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]
