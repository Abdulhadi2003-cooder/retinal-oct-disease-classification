# retinal-oct-disease-classification
Deep learning models for retinal disease classification using OCT images
# Retinal OCT Disease Classification using Deep Learning

This project presents an end-to-end deep learning pipeline for automated
classification of retinal diseases using Optical Coherence Tomography (OCT)
images. The goal is to assist in early diagnosis of retinal conditions by
leveraging convolutional neural networks and modern AI techniques.

## Dataset
The project uses the publicly available OCT2017 (Kermany) dataset, which
contains labeled retinal OCT images across multiple disease categories.

Dataset link:
https://www.kaggle.com/datasets/paultimothymooney/kermany2018

## Methods and Models
The following approaches were implemented and compared:

- Transfer learning using VGG16 pretrained on ImageNet
- A custom-designed CNN architecture
- Patch-based Multiple Instance Learning (MIL) with Top-K pooling
- Contrast enhancement using CLAHE
- Duplicate image removal using MD5 hashing
- Hyperparameter optimization using Optuna

## Results
- VGG16 achieved up to 99.48% classification accuracy
- Custom CNN achieved up to 97.11% accuracy
- Patch-based MIL model achieved approximately 98% accuracy with improved
  interpretability

## Graphical User Interface (GUI)
An interactive GUI was developed using Gradio to allow real-time testing of
new OCT images. The GUI loads pretrained models and visualizes predictions.

## How to Run the Project
1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/paultimothymooney/kermany2018

2. Edit the dataset path in `Remove_duplicates.ipynb` (first cell).

3. Run the notebook once to identify and move duplicate images into a
   separate folder.

4. Update the path of the cleaned dataset in the remaining notebooks
   (first or second cells).

5. Use the following notebooks to train and evaluate models:
   - `own_cnn_model.ipynb`
   - `patch_model.ipynb`
   - `transfer_model.ipynb`

6. Run `gui.ipynb` to launch the GUI for testing new images.
   (Pretrained `.keras` model weights are used for inference.)

## Technologies Used
Python, TensorFlow/Keras, Convolutional Neural Networks (CNNs),
Transfer Learning, Multiple Instance Learning (MIL),
Optuna, Gradio
