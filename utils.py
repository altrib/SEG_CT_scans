import torch
import torchvision
from dataset import CTScanData
from torch.utils.data import DataLoader
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from tqdm import tqdm

from lossAndMetrics import *



def save_checkpoint(state, file_name = "my_checkpoint.pth.tar"):
    print("=> save checkpoint")
    torch.save(state, file_name)

def load_checkpoint(checkpoint, model):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=2,
        pin_memory=True,
        index = 200
):
    train_ds = CTScanData(
        train_dir, 
        train_maskdir, 
        transform = train_transform,
        index = index
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_ds = CTScanData(
        val_dir, 
        val_maskdir, 
        transform = val_transform,
        index = 500
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader



def check_accuracy(loader, model,num_classes, device="cuda"):

    model.eval()
    t = time.time()
    print("---check accuracy---")

    preds = []
    targets = []
    with torch.no_grad():
        for (data,target) in tqdm(loader):
            data = data.unsqueeze(1).float().to(device)
            pred = torch.argmax(model(data),dim=1).int()
            for i in range(2):
                preds.append(pred[i].to(device).flatten())
                targets.append(target[i].to(device).flatten())
            
        preds = torch.stack(preds)
        targets = torch.stack(targets)
        # print(preds.shape)
        # print(targets.shape)
        # print(time.time() - t)
        pa = pixel_accuracy(preds,targets)
        # print(time.time() - t)
        # dc = dice_coefficient(preds,targets,num_classes)
        # print(time.time() - t)
        # miou = mean_iou(preds,targets,num_classes)
        # print(time.time() - t)
        ari = adjusted_rand_index_score(preds.to("cpu").numpy(),targets.to("cpu").numpy())
        # print(time.time() - t)
        

    score_dict = pd.DataFrame({
        'pixel accuracy': [pa.item()],
        # 'dice coefficient': [dc.item()],
        # 'mean IoU': [miou.item()],
        'adjusted rand index': [ari]
        })

    
    # print(score_dict)
    


    model.train()
    return score_dict


def plot_slice_seg(slice_image, seg,folder,idx,y=None):
    if y is None:
        fig, axes = plt.subplots(1, 2)
    else:
        fig, axes = plt.subplots(1, 3)
        axes[2].imshow(slice_image.reshape((512,512)), cmap="gray")
        y_masked = np.ma.masked_where(y.reshape((512,512)) == 0, (y.reshape((512,512))))
        axes[2].imshow(y_masked, cmap="tab20")
        plt.axis("off")
    axes[0].imshow(slice_image.reshape((512,512)), cmap="gray")
    axes[1].imshow(slice_image.reshape((512,512)), cmap="gray")
    seg_masked = np.ma.masked_where(seg.reshape((512,512)) == 0, (seg.reshape((512,512))))
    axes[1].imshow(seg_masked, cmap="tab20")
    plt.axis("off")

    plt.savefig(os.path.join(folder,f"{idx}.png"))
    plt.close(fig)




def save_prediction_as_imgs(
        loader, model, folder, device="cuda",batch_size=4,some = False
):
    model.eval()
    os.makedirs(os.path.dirname(folder), exist_ok=True)

    if some:
        idx = np.random.randint(0,len(loader.dataset),2)
        subset = torch.utils.data.Subset(loader.dataset,idx)
        subloader = DataLoader(subset, batch_size=1, shuffle=False)
        for i,(x, _) in enumerate(subloader):
                    with torch.no_grad():
                        x = x.unsqueeze(1).float().to(device)
                        preds = torch.argmax(model(x),dim=1).int().to("cpu").numpy().flatten()
                        x = x.to("cpu").numpy()
                    print(f" -> image {idx[i]}")
                    plot_slice_seg(x,preds,folder,idx[i])

    else:
        for idx, (x, y) in enumerate(loader):
            print(f"--batch {idx}--")
            with torch.no_grad():
                x = x.unsqueeze(1).float().to(device)
                preds = torch.argmax(model(x),dim=1).int().to("cpu").numpy()
                x = x.to("cpu").numpy()
                y = y.to("cpu").numpy()
            for i in range(len(preds)):
                print(f" -> image {batch_size*idx+i}")
                plot_slice_seg(x[i],preds[i],folder,batch_size*idx+i,y[i])
    model.train()



def save_predictions_to_csv(
        loader, model, path, device="cuda",batch_size=4
):
    model.eval()
    output = []
    for idx, (x, y) in enumerate(loader):
        print(f"\r--batch {idx}--")
        with torch.no_grad():
            x = x.unsqueeze(1).float().to(device)
            preds = torch.argmax(model(x),dim=1).int().to("cpu").numpy()
            for pred in preds:
                output.append(pred.flatten())
    output = np.array(output)
    output = pd.DataFrame(output).T
    output.index = output.index.map(lambda x: "Pixel " + str(x))
    output.columns = output.columns.map(lambda x: str(x)+".png")
    # output.to_csv(path)
    return output