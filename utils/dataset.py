import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FundusDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Search for image in subdirectories
        img_name = row['id_code'] + '.png'
        img_path = None
        for root, dirs, files in os.walk(self.img_dir):
            if img_name in files:
                img_path = os.path.join(root, img_name)
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in {self.img_dir}")
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert multi-class to binary: 0 = No DR, 1-4 = DR
        diagnosis = row['diagnosis']
        label = 0 if diagnosis == 0 else 1
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms(split, config):
    aug = []
    if split == 'train':
        for a in config.get('augmentation', []):
            if a == 'horizontal_flip':
                aug.append(A.HorizontalFlip(p=0.5))
            elif a == 'vertical_flip':
                aug.append(A.VerticalFlip(p=0.5))
            elif a == 'random_brightness_contrast':
                aug.append(A.RandomBrightnessContrast(p=0.5))
    aug.append(A.Resize(config['img_size'], config['img_size']))
    aug.append(A.Normalize())
    aug.append(ToTensorV2())
    return A.Compose(aug)
