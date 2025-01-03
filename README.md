# Unlearning via NGD: A Comparison with Retraining from Scratch

## Overview

This repository contains the code for an experiment evaluating a case of Algorithm 1 in the paper "Faster Machine Unlearning via Natural Gradient
Descent" by Lev and Wilson (2024). The experiment involved training an One-Layer Perceptron (OLP) model on the MNIST dataset, followed by simulating a delete request of 1000 random input data points and applying Algorithm 4 for unlearning. We compare the time and accuracy performance of retraining the model from scratch with the unlearning process using NGD (Natural Gradient Descent) as implemented in Algorithm 1 in Lev and Wilson Paper.

## Experiment Details

### Model Configuration

- **Dataset:** MNIST
- **Model:** One-Layer Perceptron (OLP)
  - **Hidden Layer:** 200 neurons
  - **Output Layer:** 10 units (for digit classification 0-9)
- **Loss Function:** Cross-Entropy Loss
- **Regularization:** L2 regularization with strength λ = 1 × 10⁻⁸
- **Optimizer:** Adam
  - Learning rate: 0.02
  - Beta values: (0.9, 0.99)
  - Epsilon: 1e-8
  - Weight decay: 10⁻⁴
- **Batch Size:** 256
- **Epochs:** 50

### NGD (Natural Gradient Descent) Configuration

- **Source:** Adapted from Shao (2021), as used in Lev and Wilson (2024)
- **Parameters:**
  - Learning rate: 0.05
  - Momentum: 0.9
  - Weight decay: 5 × 10⁻⁴
  - Batch Size: 128
  - Update-period: 4

### Procedure

1. **Training:** Train the OLP model on the MNIST dataset using the above configuration.
2. **Delete Simulation:** Simulate a delete request by randomly removing 1000 data points from the dataset.
3. **Unlearning via Algorithm 1:** Apply the NGD-based method to unlearn the deleted data.
4. **Retraining Comparison:** Compare the unlearning process against retraining the model from scratch.

### Results

Currently implementation in this project is in progress, and so no Results are provided.

## Installation

Clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/AnangaB/Machine_Unlearning_Using_NGD.git
cd Machine_Unlearning_Using_NGD
pip install -r requirements.txt
```
