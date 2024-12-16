import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload  # Python 3.4+

from model import UNET
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_prediction_as_imgs
)




# Hyperaprameters etc
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "C:/Users/alexa/Desktop/datachallenge/X_train"
TRAIN_MASK_DIR = "C:/Users/alexa/Desktop/datachallenge/Y_train.csv"
TEST_IMG_DIR = "C:/Users/alexa/Desktop/datachallenge/X_test"
TEST_MASK_DIR = "C:/Users/alexa/Desktop/datachallenge/pred"
INDEX_SUPERVISED = 2000



def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    epoch_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.unsqueeze(1).float().to(device = DEVICE)
        targets = targets.long().to(DEVICE)
    
        # foreward
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
    
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(loader)
    return avg_epoch_loss


def main():
    torch.cuda.empty_cache()
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            A.Rotate(limit=35,p=1),
            A.VerticalFlip(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean = 0,
                std = 1,
                max_pixel_value=255
            ),
            ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            A.Normalize(
                mean = 0,
                std = 1,
                max_pixel_value=255
            ),
            ToTensorV2(),
        ],
    )

    print(f"Loading data /")
    train_loader, val_loader = get_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        train_dir=TRAIN_IMG_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        train_transform=train_transform,
        val_dir=TEST_IMG_DIR,
        val_maskdir=TEST_MASK_DIR,
        val_transform=val_transform,
        index = INDEX_SUPERVISED
    )
    print(f"-----------------> Data loaded")
    
    max_classes = train_loader.dataset.get_masks().max()+1
    print(f"Max number of class : {max_classes}")

    model = UNET(in_channels = 1, out_channels = max_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scaler = torch.amp.GradScaler(device=DEVICE)

    # load_checkpoint(torch.load("./checkpoints/checkpoint-204.pth.tar"),model)

    train_losses = []
    train_scores = []
    

    # compute initial loss
    model.eval()
    train_loss = 0.0

    for images, masks in train_loader:
        with torch.no_grad():
            images = images.unsqueeze(1).float().to(DEVICE)
            masks = masks.long().to(DEVICE) 
            assert masks.min() >= 0 and masks.max() < max_classes, "Classe dans les masques hors des limites !"+str(masks.max())




            outputs = model(images)

            loss = loss_fn(outputs, masks)
            train_loss += loss.item()

    print(f"masks dim : {masks.shape}")
    print(f"images dim : {images.shape}")
    print(f"outputs dim : {outputs.shape}")

    avg_loss = train_loss / len(train_loader)
    print(f"Initial Loss: {avg_loss:.4f}")

    model.train()

    # train
    # epoch = 204
    avg_epoch_loss = avg_loss
    # while epoch < 250 and avg_epoch_loss > 0.2:
    

    for epoch in range(NUM_EPOCHS):
        
        # torch.cuda.empty_cache()

        print(f"=============================Epoch {epoch}/{NUM_EPOCHS}==========================")

        avg_epoch_loss = train_fn(loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn, scaler=scaler)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}")

        train_losses.append(avg_epoch_loss)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint,f"./checkpoints/checkpoint-{epoch}.pth.tar")

        # check accuracy
        # if ((epoch+1) % 20 == 0):
        #     train_scores = pd.concat([train_scores, check_accuracy(train_loader,model,max_classes,DEVICE)], axis=0, ignore_index=True)


        # save imgs
        save_prediction_as_imgs(
            val_loader, model, folder=f"./pred/epoch{epoch}/", batch_size=BATCH_SIZE,some=True
        )

        epoch +=1

    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o')
    plt.title("Loss curve during training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    # plt.plot(range(1, NUM_EPOCHS//20), train_scores, marker='o')
    # plt.title("Adjusted rand score during training")
    # plt.xlabel("Epochs")
    # plt.ylabel("Adjusted rand score")
    # plt.grid()
    # plt.show()

if __name__ == "__main__":
    main()