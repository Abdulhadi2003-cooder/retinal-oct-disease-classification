# retinal-oct-disease-classification
Deep learning models for retinal disease classification using OCT images
# Retinal OCT Disease Classification using Deep Learning

This project implements an end-to-end deep learning pipeline for automated
classification of retinal diseases using Optical Coherence Tomography (OCT)
images. The system aims to support early diagnosis by leveraging modern
computer vision and deep learning techniques.

## Dataset
The project uses the publicly available OCT2017 (Kermany) dataset, which
contains labeled retinal OCT images across multiple retinal disease classes.

Dataset:
https://www.kaggle.com/datasets/paultimothymooney/kermany2018

## Methods and Models
The following approaches were implemented and evaluated:

- Transfer learning using VGG16 pretrained on ImageNet
- Custom convolutional neural network (CNN)
- Patch-based Multiple Instance Learning (MIL) with Top-K pooling
- CLAHE-based contrast enhancement
- Duplicate image removal using MD5 hashing
- Hyperparameter optimization using Optuna

## Results
- VGG16 achieved up to 99.48% classification accuracy
- Custom CNN achieved up to 97.11% accuracy
- Patch-based MIL model achieved approximately 98% accuracy with improved
  interpretability

## Graphical User Interface (GUI)
A Gradio-based graphical user interface was developed to allow real-time
testing of new OCT images using pretrained models.

## How to Run
1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/paultimothymooney/kermany2018

2. Update the dataset path in `remove_duplicate_images.ipynb`.

3. Run the notebook once to identify and move duplicate images.

4. Update dataset paths in the remaining notebooks.

5. Train or evaluate models using:
   - `cnn_model.ipynb`
   - `cnn_model_optuna.ipynb`
   - `vgg16_transfer_learning.ipynb`
   - `patch_mil_model.ipynb`

6. Launch the GUI using `gradio_gui.ipynb`.

## Technologies
Python, TensorFlow/Keras, Deep Learning, CNNs, Transfer Learning,
Multiple Instance Learning (MIL), Optuna, Gradio

