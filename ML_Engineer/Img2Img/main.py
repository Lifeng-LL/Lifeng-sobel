import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from data import COCODataset
from model import SobelNet
from torch.utils.data import DataLoader
from os import makedirs
from os.path import join, exists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(dataset_path, output_path=None, num_epochs=20, batch_size=16):
    train_dataset = COCODataset(dataset_path, mode="train")
    val_dataset = COCODataset(dataset_path, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = SobelNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    train_log_frequency, val_log_frequency = 100, 20
    batch_per_epoch = len(train_loader) // batch_size

    for epoch in range(num_epochs):
        for i, (img, gt) in enumerate(train_loader):
            img = img.to(device)
            gt = gt.to(device)
            pred_train = model(img)
            loss_train = criterion(pred_train, gt)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if (batch_per_epoch * epoch + i) % train_log_frequency == 0:
                train_losses.append(loss_train.item())
        
        for j, (img, gt) in enumerate(val_loader):
            img = img.to(device)
            gt = gt.to(device)
            pred_test = model(img)
            loss_val = criterion(pred_test, gt)

            if (batch_per_epoch * epoch + j) % val_log_frequency == 0:
                val_losses.append(loss_val.item())

                if output_path is not None:
                    sobel_output = pred_test.detach().cpu().clamp(0, 1).numpy()
                    gt_out = gt.detach().cpu().numpy()
                    for k in range(sobel_output.shape[0]):
                        cv2.imwrite(join(output_path, f"{epoch}_{j}_{k}_sobel.jpg"), (sobel_output[k, 0].squeeze() * 255).astype(np.uint8))
                        cv2.imwrite(join(output_path, f"{epoch}_{j}_{k}_gt.jpg"), (gt_out[k].squeeze() * 255).astype(np.uint8))


if __name__ == "__main__":
    dataset_path = "/home/lifeng/Downloads/val2017"
    output_path = "/home/lifeng/sobel/output"
    if not exists(output_path):
        makedirs(output_path)

    run(dataset_path, output_path)
    print("done")
