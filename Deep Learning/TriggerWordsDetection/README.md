# Trigger Words Detection: Deep Learning for Audio Wake-Word Detection

This project implements a complete pipeline for wake-word (trigger word) detection, covering everything from synthetic data generation to training high-performance sequence models using GRUs and Transformers.

## Project Overview
Detecting a specific word in an audio stream requires robust sequence modeling and significant amounts of varied data. This project solves the data scarcity problem through automated synthesis and compares different state-of-the-art architectures.

## Key Components

### 1. Data Engineering & Synthesis
Since real-world wake-word data is hard to collect in volume, this project includes a sophisticated synthesis engine:
- **Augmentation**: Uses `audiomentations` for pitch shifting, time stretching, and shifting.
- **Mixing**: Randomly overlays positive words and negative words onto varied background noises (rain, crowd, white noise).
- **Feature Extraction**: Converts raw audio waveforms into **Spectrograms**, allowing the models to learn from frequency-domain representations.
- **Tools**: `librosa`, `pydub`, `soundfile`.

### 2. Model Architectures
Two primary approaches are implemented:
- **Recurrent Neural Network (GRU)**: A Gated Recurrent Unit architecture optimized for real-time sequence processing.
- **Attention/Transformer**: A more modern approach using self-attention mechanisms to capture long-range dependencies in audio sequences more effectively.

### 3. Hyperparameter Optimization
Uses `Keras Tuner` with **Bayesian Optimization** to systematically find the best:
- Number of units in recurrent layers.
- Dropout rates.
- Learning rates and batch sizes.

## Installation & Dependencies
```bash
pip install tensorflow keras-tuner librosa pydub audiomentations soundfile matplotlib
```

## Project Structure
- `dataset_creation.ipynb`: The synthesis engine to generate training data.
- `training_eval.ipynb`: Training pipeline for the GRU-based model.
- `training_eval_attention.ipynb`: Training pipeline for the Transformer/Attention model.

## Results
The synthesis pipeline enables the creation of thousands of unique training samples from just a few recording of the trigger word, resulting in a model that generalizes well to different voices and noisy environments.
