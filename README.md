# Rock–Paper–Scissors CNN

Image classification of hand gestures (rock, paper, scissors) using three convolutional neural networks of increasing complexity, built in Python with Keras. The final, hyperparameter-tuned model reaches **98.5% accuracy** on a held-out test set.

## Overview

The goal is to design, train and compare CNNs that recognise the three gestures, balancing accuracy, generalisation and computational cost under limited data and hardware. The project covers:

- a stratified train / validation / test split with a fixed seed for reproducibility
- three data-augmentation strategies matched to model complexity
- three CNN architectures (Tiny, Small, Medium)
- hyperparameter tuning with 3-fold cross-validation
- evaluation with accuracy, precision, recall, F1-score and confusion matrices
- a generalisation test on 12 personal photos taken outside the dataset

## Dataset

Rock–Paper–Scissors image dataset: photos of hands on a green background, three roughly balanced classes. https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

- Images resized to 128×128 and scaled to [0, 1]
- Stratified split: 70% train, 15% validation, 15% test (seed = 42)

## Models

All networks use 3×3 convolutions with ReLU, max pooling, global average pooling, dropout and a 3-unit softmax output. They are trained with Adam and categorical cross-entropy with label smoothing.

| Model | Parameters | Conv layers | Dropout | Augmentation |
|---|---|---|---|---|
| Tiny CNN | 5,187 | 2 | 0.2 | Basic (flip, small rotation, zoom, contrast) |
| Small CNN | 28,835 | 3 | 0.3 | RandAugment (KerasCV, N=2, M=0.3) |
| Medium CNN | 139,811 | 5 | 0.4 (tuned) | "Green strong": green-background removal + strong photometric and geometric changes |

The Medium CNN was tuned with 3-fold cross-validation over learning rate (0.0005, 0.001), dropout (0.4, 0.5) and batch size (32). The best configuration (learning rate 0.001, dropout 0.4, batch size 32) was retrained on train + validation for 15 epochs and evaluated once on the untouched test set.

## Results

Held-out test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Tiny CNN | 0.589 | 0.610 | 0.589 | 0.563 |
| Small CNN | 0.846 | 0.847 | 0.846 | 0.845 |
| Medium CNN | **0.985** | **0.985** | **0.985** | **0.985** |

Generalisation test: the Medium CNN correctly classified **10 out of 12** personal photos (~83%). Both errors were rock predicted as scissors, mostly when the hand occupied a small part of the frame.

## Key takeaways

- Model capacity and augmentation both matter: F1 goes from 0.563 to 0.845 to 0.985 across the three models.
- Augmentation designed for the data (removing the green background) helps the model focus on hand shape instead of background.
- The test set comes from the same distribution as the training data, so 98.5% is an optimistic estimate. The 12-photo test is very small, but shows that scale and framing are the main weaknesses.
  

## Repository structure

```
configs/        experiment configurations
src/            source code (data preparation, models, training, evaluation)
report.pdf      full project report with curves, confusion matrices and error analysis
requirements.txt
```

## Tech stack

Tensorflow, Numpy, Pandas, Matplotlib, scikit-learn, tqdm, jsonschema, keras-tuner, pillow, pyyam

