from tqdm import tqdm
import torch
import csv
import numpy as np
from loss import IOU

def Train(model, epochs, optimizer, scheduler, loss_fn, train_iter, val_iter, device, early_stopping, DeepLab=False):
    # Create dictionary to store history
    Loss = {"val_iou":[]}

    with open('./models/readme.txt', 'w') as f:
        f.write('Model: {} \n Max Epochs: {} \n Optimizer: {} \n Scheduler: {} \n Loss Function: {} \n Device: {} \n Early Stopping: {}'.format(model, epochs, optimizer, scheduler, loss_fn, device, early_stopping))

    with open('./models/log.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","train_iou","val_loss","val_iou"])
    
    # Set patience to zero.
    patience = 0

    for epoch in range(epochs):
        # Set model in training mode
        model.train()

        # Initialise cumulative loss
        train_loss, train_iou, val_loss, val_iou = 0, 0, 0, 0
        
        # Print LR if it has decreased.
        if epoch != 0:
            if optimizer.param_groups[0]['lr'] < LR:
                print('Learning rate decreased to ', optimizer.param_groups[0]['lr'])
        else:
            print('Initial learning rate set to ', optimizer.param_groups[0]['lr'])
        LR = optimizer.param_groups[0]['lr']

        # Loop over the training set
        for i, data in enumerate(tqdm(train_iter)):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero previous gradients
            optimizer.zero_grad()
            
            
            # Generate predictions and loss with current model parameters
            if DeepLab == True:
                outputs = model(inputs)["out"]
            else:
                outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Initiate backpropagation to adjust loss weights
            loss.backward()
            optimizer.step()

            # Update total training loss
            train_loss += loss
            train_iou += IOU(outputs, labels, device)
        train_steps = i+1

        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()

            # Loop over the validation set
            for i, data in enumerate(tqdm(val_iter)):
                inputs, labels = data[0].to(device), data[1].to(device)

                # Calculate validation loss
                if DeepLab == True:
                    outputs = model(inputs)["out"]
                else:
                    outputs = model(inputs)
                val_loss += loss_fn(outputs, labels)
                val_iou += IOU(outputs, labels, device)
            val_steps = i+1
        
        # Calculate the average training and validation loss
        avg_train_loss = float(train_loss / train_steps)
        avg_train_iou = float(train_iou / train_steps)
        avg_val_loss = float(val_loss / val_steps)
        avg_val_iou = float(val_iou / val_steps)
        
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Save the best model if appropriate, else continue.
        if epoch == 0:
            torch.save(model.state_dict(), "./models/model.pth")
            print("Saved best model!")
        elif avg_val_iou > np.max(Loss["val_iou"]):
            torch.save(model.state_dict(), './models/model.pth')
            print("Saved best model!")
            patience = 0
        else:
            patience += 1

        # Update train and val loss history
        Loss["val_iou"].append(avg_val_iou)

        with open('./models/log.csv', 'a') as csv_file:
            dict_object = csv.DictWriter(csv_file, fieldnames=["epoch","train_loss","train_iou","val_loss","val_iou"])
            dict_object.writerow({"epoch":epoch,"train_loss":avg_train_loss,"train_iou":avg_train_iou,"val_loss":avg_val_loss,"val_iou":avg_val_iou})

        print("Epoch {}, Train Loss {:3f}, Train IOU {:3f}, Val Loss {:3f}, Val IOU {:3f}".format(
            epoch, avg_train_loss, avg_train_iou, avg_val_loss, avg_val_iou))
        
        if patience > early_stopping:
            print("Early stopping triggered, best val IOU: {}".format(np.max(Loss["val_iou"])))
            break