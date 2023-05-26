import cv2
import torch
from torch.utils.data import Dataset

class DFUDataset(Dataset):
    """ Custom dataset initiation for diabetic foot ulcer segmentation dataset. """ 

    def __init__(self, image_dir, mask_dir, transform=None):
        self.transform = transform
        self.images, self.masks = image_dir, mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = cv2.imread(self.images[idx]), cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'].to(torch.float32) / 255, transformed['mask'].to(torch.float32) / 255

        return image, mask

class TestDataset(Dataset):
    """ Custom dataset initiation for diabetic foot ulcer segmentation dataset. """ 

    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.images = image_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image'].to(torch.float32) / 255

        return image