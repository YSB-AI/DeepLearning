# Deep Learning Portfolio

Welcome to my Deep Learning portfolio! This repository showcases a variety of projects spanning computer vision, natural language processing, and sequence modeling. Each project demonstrates different architectural patterns, from foundational RNNs and CNNs to modern Transformers and automated hyperparameter optimization.

## 🚀 Key Projects

### [Trigger Words Detection](./TriggerWordsDetection/)
A complete pipeline for audio wake-word detection.
- **Task**: Detect a specific word within an audio stream.
- **Approach**: Synthesized a large-scale dataset with noise augmentation. Compared **GRU-based** models with modern **Attention/Transformer** architectures.
- **Key Tech**: `TensorFlow`, `Keras Tuner` (Bayesian Optimization), `librosa`, `pydub`, `audiomentations`.

### [Sentiment Classification](./Sentiment%20Classification/)
A comparative study of statistical and transformer-based sentiment analysis.
- **Task**: Classify movie review sentiment (Positive/Negative).
- **Approach**: Comparative analysis between a classic **Naive Bayes** log-likelihood model and a fine-tuned **BERT** (Small BERT) model.
- **Discovery**: In this specific dataset context, the statistical model achieved superior performance (AUC: 0.88 vs 0.84) with significantly lower computational cost.
- **Key Tech**: `TensorFlow Hub` (BERT), `scikit-learn`, `Bayesian Optimization`.

### [Digit Classification](./DigitClassification/)
Robust handwritten digit recognition with automated tuning.
- **Task**: Classify MNIST digits (0-9).
- **Approach**: Implemented a **CNN** with automated hyperparameter search (filters, kernel sizes, dropout) using **Bayesian Optimization**. Includes detailed error analysis via confusion matrices and visualization.
- **Key Tech**: `Keras Tuner`, `Seaborn` (Visualizations), `TensorFlow`.

---

## 📚 [Classic Approach](./Classic%20Approach/)
This directory serves as the bedrock of the portfolio, focusing on the fundamental building blocks of Deep Learning.

- **Digit Classification**: Foundational multi-layer perceptrons and basic CNNs. Use the Classic version to see how raw data is processed and transformed, and use the Root version when you want to see how to bake that into a high-accuracy, optimized model.
- **Sentiment & Emojify**: Text-to-emoji mapping using word embeddings and RNNs.
- **Literary Text Generation**: Character-level RNNs trained on literary corpora to learn stylistic language patterns.
- **Basic NMT**: Introduction to sequence-to-sequence translation.

---

## 🛠️ Main Technologies & Tools
Across these projects, I leverage a modern deep learning stack:
- **Frameworks**: TensorFlow / Keras
- **Optimization**: Keras Tuner (Bayesian Optimization, Hyperband)
- **Audio Processing**: Librosa, Pydub, Soundfile
- **NLP**: BERT (via TF Hub), Word Embeddings (Word2Vec/GloVe concepts)
- **Visualization**: Matplotlib, Seaborn

## 📈 Learning Objectives
- Architecting and training **CNNs**, **RNNs (GRU/LSTM)**, and **Transformers**.
- Implementing automated **Hyperparameter Tuning** for model optimization.
- Performing rigorous **Error Analysis** and performance evaluation.
- Building sophisticated **Data Synthesis** and **Augmentation** pipelines.
