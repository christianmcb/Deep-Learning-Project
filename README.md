# Deep-Learning-Project
This project implements U-Net and DeepLabV3_MobileNetV3 deep learning models for image segmentation specific to the Diabetic Foot Ulcer dataset provided on https://dfuc2022.grand-challenge.org.

These models have been built using Python version 3.9.16 and PyTorch version 1.12.1.

**Directory Guidance**

  - [**Assignment.xlsx**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/Assignment.xlsx) shows the results of the model training tasks based on experiments.

Notebooks:
  - [**main.ipynb**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/notebooks/main.ipynb) shows the full building and running to produce best model.
  - [**augmentation.ipynb**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/notebooks/augmentation.ipynb) gives an overview of the different image augmentation methods used in this project.
  - [**exploration.ipynb**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/notebooks/exploration.ipynb) explores the dataset and gives an overview of statistics.
  - [**modelTesting.ipynb**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/notebooks/modelTesting.ipynb) evaluates the baseline and best models that have been trialled.
  - [**train_test_split.ipynb**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/notebooks/train_test_split.ipynb) shows an example of how the images can be split into training and testing lists.
  - **csv files** save of the descriptive statistics from the exploration.ipynb

Code - OOP Programming of the Project:
  - [**main.py**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/code/main.py) running this file will train the model and produce results saved in "./models"
  - [**loss.py**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/code/loss.py) defines the dice loss function, the IOU evaluation metric and loads BCE with Logits Loss from Pytorch.
  - [**dataset.py**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/code/dataset.py) the torch dataset for loading and normalising images.
  - [**model.py**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/code/dataset.py) builds a U-Net convolutional neural network and load DeepLabV3_MobileNetV3 from PyTorch.
  - [**optimiser.py**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/code/optimiser.py) loads the SGD and ADAM optimisers from PyTorch.
  - [**readFiles.py**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/code/readFiles.py) reads the directories of the images saved in "./dfuc2022/"
  - [**training.py**](https://github.com/christianmcb/Deep-Learning-Project/blob/main/code/training.py) provides the training loop for training and evaluation of validation data.


**Example Model Save:**
Shows an example of how the model and log will be saved when the main.py or main.ipynb files have been run.
