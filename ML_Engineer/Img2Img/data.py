import numpy as np
import torch
import cv2

from PIL import Image
from natsort import natsorted
from glob import glob
from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class COCODataset(Dataset):
    def __init__(self, dataset_path, mode="train", target_size=(640, 480)):
        self.img_paths = natsorted(glob(join(dataset_path, mode, "*.jpg")))
        self.target_size = target_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = np.array(img).astype(np.uint8)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w, _ = img.shape
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        gt_x = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3).astype(np.float32) / 255.0
        gt_y = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3).astype(np.float32) / 255.0
        gt = cv2.addWeighted(gt_x, 0.5, gt_y, 0.5, 0)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0), np.expand_dims(gt, axis=0)
