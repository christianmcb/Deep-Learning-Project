# Deep-Learning-Project
This project implements U-Net and DeepLabV3_MobileNetV3 deep learning models for image segmentation specific to the Diabetic Foot Ulcer dataset provided on https://dfuc2022.grand-challenge.org.

These models have been built using Python version 3.9.16 and PyTorch version 1.12.1.

**Directory Guidance**
Notebooks:
  - **main.ipynb** shows the full building and running to produce best model.
  - **augmentation.ipynb** gives an overview of the different image augmentation methods used in this project.
  - **exploration.ipynb** explores the dataset and gives an overview of statistics.
  - **modelTesting.ipynb** evaluates the baseline and best models that have been trialled.
  - **train_test_split.ipynb** shows an example of how the images can be split into training and testing lists.
  - **csv files** save of the descriptive statistics from the exploration.ipynb

Code - OOP Programming of the Project:
  - **main.py** running this file will train the model and produce results saved in "./models"
  - **dataset.py** the torch dataset for loading and normalising images.
  - **model.py** builds a U-Net convolutional neural network and load DeepLabV3_MobileNetV3 from PyTorch.
  - **optimiser.py** loads the SGD and ADAM optimisers from PyTorch.
  - **readFiles.py** reads the directories of the images saved in "./dfuc2022/"
  - **training.py** provides the training loop for training and evaluation of validation data.
