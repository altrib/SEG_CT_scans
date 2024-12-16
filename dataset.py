import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class CTScanData(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None,index=200):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list(sorted(Path(data_dir).glob("*.png"), key=lambda filename: int(filename.name.rstrip(".png"))))[:index]
        self.masks =  pd.read_csv(mask_dir, index_col=0).T.values[:index] if os.path.isfile(mask_dir) else np.zeros((len(self.images),512*512)) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(cv2.imread(str(self.images[index]), cv2.IMREAD_GRAYSCALE))
        return image, self.masks[index].reshape((512,512))
    
    def get_masks(self):
        return self.masks
    
    def get_images(self):
        images_list = []
        for image in self.images:
            images_list.append(np.array(cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)))
        return np.stack(images_list, axis=1)
