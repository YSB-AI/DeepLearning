# Digit Classification with CNN and Bayesian Optimization

This project implements a robust handwritten digit classification system using the MNIST dataset. It goes beyond a simple baseline by incorporating automated hyperparameter tuning and detailed error analysis.

## Project Overview
The goal is to classify images of handwritten digits (0-9) using a Convolutional Neural Network (CNN). The project focuses on optimizing model performance through systematic hyperparameter search and identifying specific patterns of misclassification.

## Key Features
- **Convolutional Neural Network (CNN)**: Utilizes the power of spatial hierarchies in image data.
- **Automated Hyperparameter Tuning**: Uses `Keras Tuner` with `BayesianOptimization` to find the best number of filters, kernel sizes, and learning rates.
- **Model Checkpointing**: Automatically saves the best model during training.
- **Error Analysis**: Detailed evaluation using confusion matrices and visualization of misclassified samples to guide future improvements.

## Model Architecture
The network consists of:
1. **Convolutional Layers**: Multiple `Conv2D` layers for feature extraction.
2. **Pooling Layers**: `MaxPooling2D` to reduce dimensionality and provide translation invariance.
3. **Regularization**: `Dropout` layers to prevent overfitting.
4. **Dense Layers**: Fully connected layers for the final classification.

## Installation & Dependencies
To run the notebooks, you will need:
```bash
pip install tensorflow keras-tuner numpy matplotlib seaborn scikit-learn
```

## How to Run
1. Navigate to the `DigitClassification` directory.
2. Open `training_eval_tuning.ipynb` for the full pipeline including hyperparameter search.
3. Open `training_eval.ipynb` for a more straightforward training and evaluation process.

## Results
The model achieves high accuracy on the MNIST test set (typically >99%). The error analysis reveals that most mistakes occur between visually similar digits like '4' and '9' or '2' and '7'.
