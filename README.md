
# Self-Pruning Neural Network

**Tredence AI Engineering Intern Case Study**

---

## Overview

This project implements a **self-pruning feed-forward neural network** trained on the CIFAR-10 dataset.

Unlike traditional pruning methods that remove weights after training, this model learns to **prune itself during training** using a learnable gating mechanism. The network automatically identifies and suppresses unnecessary connections, resulting in a sparse and efficient model.

---

## Core Idea

Each weight `w` in the network is associated with a learnable gate `g`:

* Gate computation:
  `g = sigmoid(gate_score)`

* Effective weight:
  `w' = w × g`

Interpretation:

* If `g ≈ 1` → weight is important
* If `g ≈ 0` → weight is pruned

---

## Model Architecture

The model is a fully connected neural network using custom prunable layers:

* Input: 32×32×3 images (flattened)
* Layers:

  * FC1: 3072 → 256 (PrunableLinear)
  * FC2: 256 → 128 (PrunableLinear)
  * FC3: 128 → 10 (PrunableLinear)

Each `PrunableLinear` layer contains:

* Standard weights and bias
* Learnable gate scores
* Sigmoid-based gating mechanism

---

## Loss Function

The model is trained using a combined loss:

Total Loss = CrossEntropyLoss + λ × SparsityLoss

Where:

* SparsityLoss = L1 norm (sum) of all gate values
* λ (lambda) controls the strength of pruning

### Why L1 Regularization?

L1 regularization encourages many gate values to become exactly zero, which leads to effective sparsity in the network.

---

## Experiments

The model is trained with multiple λ values to study the trade-off between accuracy and sparsity:

* λ = 0.1 (weak pruning)
* λ = 0.5 (balanced pruning)
* λ = 2.0 (strong pruning)
* λ = 5.0 (aggressive pruning)

---

## Results

| Lambda | Test Accuracy | Sparsity Level |
| ------ | ------------- | -------------- |
| 0.1    | High          | Low            |
| 0.5    | Good          | Moderate       |
| 2.0    | Reduced       | High           |
| 5.0    | Low           | Very High      |

---

## Key Observations

* Increasing λ increases sparsity but reduces accuracy
* Very small λ does not prune effectively
* Very large λ over-prunes and hurts performance
* λ = 0.5 provides the best balance between accuracy and sparsity

---

## Gate Distribution Analysis

The best model (λ = 0.5) shows a clear bimodal distribution:

* A spike near 0 → pruned weights
* A cluster away from 0 → important weights

This confirms that the network successfully learns to distinguish between useful and unnecessary parameters.

---

## Best Model

The model trained with λ = 0.5 is selected as the final model because it:

* Maintains strong classification accuracy
* Achieves meaningful sparsity
* Demonstrates clear self-pruning behavior

---

## How to Run

```bash
pip install torch torchvision matplotlib
jupyter notebook SelfPruningNetwork_Final.ipynb
```

---

## Features

* Custom PrunableLinear layer
* Learnable gating mechanism
* L1-based sparsity regularization
* Multi-lambda experimentation
* Accuracy vs sparsity analysis
* Gate distribution visualization

---

## Conclusion

This project demonstrates that neural networks can:

* Learn to prune unnecessary connections during training
* Achieve model compression without post-processing
* Maintain performance with fewer active parameters


---

If you want, I can also give you:

* 🔥 2–3 line **GitHub description (very important for shortlist)**
* 🎯 **perfect commit message + repo structure**
* 💬 **interview explanation (they WILL ask this)**
