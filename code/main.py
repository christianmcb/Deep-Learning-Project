import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dataset import DFUDataset
from loss import DiceLoss, BCEWithLogitsLoss
from optimiser import SGD, Adam
from model import UNet, DeepLabV3_MobileNet_V3_Large
from readFiles import ReadFiles
from training import Train

def Configurations():
    """ 
    Set the batch size for the data loader, large batch sizes can run into problems with memory.
    Suggested batch sizes: 2, 4, 8, 16
    """
    batch_size = 16

    """
    Set the maximum number of epochs for the training phase, default setting of 1000. Model is likely
    to converge before 1000 epochs when early stopping is set.
    """
    max_epochs = 1000

    """
    Set the patience for early stopping, defualt 15. The training phase will end once the validation IOU
    has not increased for the number of epochs set in early stopping.
    """
    early_stopping = 15

    """
    Set the model architecture for deep learning.
    Options: 
    'DeepLabModel()'
    'UNet()'
    'DeepLabV3_MobileNet_V3_Large()'
    """
    model = DeepLabV3_MobileNet_V3_Large()
    
    #model = DeepLabV3_MobileNet_V3_Large()
    DeepLab = True

    """
    Set the model optimizer for training the selected model.
    Options:
    'SGD(model)'
    'Adam(model)'
    """
    if batch_size == 2:
        optimizer = Adam(model, lr=0.0001)
    elif batch_size == 16:
        optimizer = Adam(model, lr=0.0005)
    else:
        optimizer = Adam(model, lr=0.0001)

    """
    Set the learning rate scheduler for the optimization function. The default selection is
    'ReduceLROnPlateau' which decreases the learning rate by a factor of 10 when val IOU has not
    increased for a patience of 5 epochs
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=5, cooldown=5, min_lr=1e-6)
    #scheduler = None

    """
    Set the loss function for training the model, default Dice.
    Options:
    'DiceLoss()'
    'BCEWithLogitsLoss()'
    """
    loss_fn = DiceLoss()

    return batch_size, max_epochs, early_stopping, model, optimizer, scheduler, loss_fn, DeepLab

def ReadDirectories():
    print("Reading files from directories.")
    data_path = ""
    # Set training directory
    train_dir = os.path.join(data_path, "dfuc2022/train/images")
    train_mask_dir = os.path.join(data_path, "dfuc2022//train/masks")

    # Set validation directory
    val_dir = os.path.join(data_path, "dfuc2022/val/images")
    val_mask_dir = os.path.join(data_path, "dfuc2022/val/masks")

    # Set test directory
    test_dir = os.path.join(data_path, "dfuc2022/test/")

    train_files = ReadFiles(train_dir)
    train_masks = ReadFiles(train_mask_dir)

    val_files = ReadFiles(val_dir)
    val_masks = ReadFiles(val_mask_dir)

    test_files = ReadFiles(test_dir)
    print("Complete.")

    return train_files, train_masks, val_files, val_masks, test_files

def Transforms():
    transform_train = A.Compose([
        #A.augmentations.dropout.cutout.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=0.2),
        #ToTensorV2(),
        #A.augmentations.transforms.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        A.augmentations.geometric.transforms.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, p=0.5),
        A.augmentations.geometric.rotate.Rotate (limit=90, p=0.5),
        A.augmentations.crops.transforms.CropAndPad (percent=(-0.3, 0.05), p=0.5),
        ToTensorV2(),
    ])

    transform_test = A.Compose([
        ToTensorV2(),
    ])

    return transform_train, transform_test

def main():
    # Read directories into file lists.
    train_files, train_masks, val_files, val_masks, test_files = ReadDirectories()

    # Get transformations from Transforms()
    transform_train, transform_test = Transforms()
    
    # Generate manual seed for reproducability
    torch.manual_seed(42)

    # Load dataset
    print("Loading datasets...")
    train_ds = DFUDataset(train_files, train_masks, transform=transform_train)
    val_ds = DFUDataset(val_files, val_masks, transform=transform_test)
    print("Complete.")

    # Initiate DataLoader
    batch_size, max_epochs, early_stopping, model, optimizer, scheduler, loss_fn, DeepLab = Configurations()
    train_iter = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
    val_iter = DataLoader(val_ds, batch_size, shuffle=True, pin_memory=True)
    print("Dataloader Initiated.")

    # Set device for running model
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Running model on device {}".format(device))

    # Initiate segmentation model for training
    model = model.to(device)

    # Model Training
    Train(model, max_epochs, optimizer, scheduler, loss_fn, train_iter, val_iter, device, early_stopping, DeepLab)


if __name__ == '__main__':
    main()