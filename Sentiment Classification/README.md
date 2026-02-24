# Sentiment Classification: Naive Bayes vs. BERT

This project explores two distinct approaches to sentiment analysis: a classic word-based statistical model (Naive Bayes) and a modern transformer-based model (BERT). Surprisingly, on this specific dataset, the simpler Naive Bayes model demonstrated superior performance.

## Project Overview

The goal is to classify movie reviews into positive or negative sentiments. The project is implemented in two main notebooks:
1. `Naive sentiment analysis model.ipynb`: Focuses on a statistical log-likelihood approach.
2. `Pretrained embeddings sentiment analysis model.ipynb`: Utilizes a BERT-based architecture with fine-tuning.

## Model Architectures

### 1. Naive Bayes (Classic Approach)
- **Logic**: A log-likelihood word-based statistical model.
- **Mechanism**: Predicts sentiment by summing the log-ratios of word frequencies across positive and negative classes.
- **Preprocessing**: Includes data filtering, stop-word removal, and word frequency thresholding.

### 2. BERT-based Model (Transformer Approach)
- **Backbone**: `bert-en-uncased-l-10-h-128-a-2` (Small BERT from TensorFlow Hub).
- **Layers**:
  - BERT Transformer layer (10 blocks, hidden size 128).
  - Global Pooled Output.
  - Optional dense layers (configured via hyperparameter tuning).
  - Output Sigmoid layer for binary classification.
- **Optimization**: Fine-tuned using Bayesian Optimization for hyperparameter selection.

## Performance Metrics

The models were evaluated on a consistent test set, yielding the following results:

| Metric | Naive Bayes | BERT-based (Small BERT) |
| :--- | :---: | :---: |
| **AUC** | **0.8848** | 0.8404 |
| **Precision** | **0.9149** | 0.8701 |
| **Recall** | **0.8839** | 0.8609 |

### Confusion Matrix (Naive Bayes)
```
[[10691,   947],
 [ 1337, 10176]]
```

## Key Findings

- **Efficiency vs. Accuracy**: The Naive Bayes model is significantly faster to train and evaluate while outperforming the small BERT variant in this specific context.
- **Data Sensitivity**: Preprocessing steps like sentence trimming (based on IQR) and handling missing words were crucial for the BERT model's stability.
- **Model Size**: The small BERT variant used (h=128) provides a balance between computational cost and performance, but remains less effective than the pure statistical approach for this particular sentiment task.

## Repository Structure

- `Naive sentiment analysis model.ipynb`: Implementation of the statistical model.
- `Pretrained embeddings sentiment analysis model.ipynb`: Implementation of the transformer-based model.
- `checkpoints/`: Saved weights for the BERT model.
- `logs/`: TensorBoard logs for hyperparameter tuning sessions.
