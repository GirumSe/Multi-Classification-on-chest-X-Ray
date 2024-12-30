# Multi-Classification on Chest X-ray Images

This repository contains a project for multi-classification of chest X-ray images using convolutional neural networks (CNNs). The model identifies 14 primary classes of chest conditions, based on preprocessed and annotated datasets.
## Group Members of **Group 7**
- Rahel Zeleke ........ Ugr/22633/13
- Lidya Gezahegn .. Ugr/23809/13
- Girum Senay ........ Ugr/22635/13
- Bereket Sahle....... Ugr/22992/13
- Biruk Maedot ...... Ugr/22845/13


## Overview
This project leverages:
- **Data preprocessing**: Includes path mapping, label encoding, and one-hot encoding.
- **Data augmentation**: Enhances the dataset using transformations to increase model generalization.
- **Model training**: Implements a CNN for classification, with layers for convolution, pooling, dropout, and dense connections.
- **Evaluation**: Generates ROC curves to measure performance across classes.

## Features
- Support for labeled image datasets stored in directory structures.
- Integrated model training and evaluation pipeline.
- Visualization of results and metrics, including ROC curves.

## Getting Started
Follow these steps to run the project:

### 1. Dataset
Ensure you have access to the chest X-ray dataset. This notebook assumes the dataset is organized as follows:
```
../input/
  images_folder/
    images/
      *.png
```
The image files are structured with relevant labels available in a CSV file.

### 2. Running the Notebook
This project is designed to be run on Kaggle.

#### Kaggle Link
Click [here](https://www.kaggle.com/code/girumsenay/multi-classification-on-chest-x-ray) to open the notebook on Kaggle. (Replace with actual Kaggle notebook link once uploaded.)

#### Steps on Kaggle
1. Upload the dataset as a Kaggle dataset or connect to an existing dataset.
2. Open the linked notebook above.
3. Run all cells sequentially to preprocess the data, train the model, and evaluate its performance.

### 3. Prerequisites
Ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `opencv-python`
- `scikit-learn`
- `keras`

These libraries are pre-installed in Kaggle environments.

## Code Walkthrough
- **Data Preprocessing**: Maps image paths to labels and applies one-hot encoding for the 14 primary chest conditions.
- **Data Augmentation**: Uses Keras' `ImageDataGenerator` for transformations such as rotations, shifts, and flips.
- **Model Definition**: Constructs a CNN with convolutional, pooling, dropout, and dense layers.
- **Training**: Trains the model using augmented training data and validates against a test set.
- **Evaluation**: Displays metrics, including ROC curves for each class.
- **Prediction**: Predicts conditions for a sample image and visualizes the result compared to the ground truth.

## Results
- The model generates probabilities for each condition.
- Results are visualized using bar charts and ROC curves.

## How to Predict on New Images
1. Replace the `sample_image_path` variable with the path to your image.
2. Run the prediction and visualization cells in the notebook.

## Future Improvements
- Fine-tune the model using hyperparameter optimization.
- Explore additional data augmentation techniques.
- Incorporate transfer learning with pre-trained models for improved accuracy.

## Acknowledgments
- Dataset: Provided by [Source of Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data).
- Frameworks: TensorFlow, Keras, and Scikit-learn.

