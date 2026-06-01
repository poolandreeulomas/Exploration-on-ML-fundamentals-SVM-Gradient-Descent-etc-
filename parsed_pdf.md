# Exercise 2 — Kernels, Gradient Descent

## Problem 2.1 (25%)

We examine two variants of gradient descent for a dataset  
T_train = { (x^(i), y^(i)) } for i = 0,...,N−1.

### Batch Gradient Descent (BGD)
The loss is evaluated over the entire dataset:

w^(k) ← w^(k−1) − α ∇_w J(w^(k−1))

J(w) = (1/N) ∑_{i=0}^{N−1} L(w, ŷ^(i), y^(i))

---

### Stochastic Gradient Descent (SGD)
Gradients are computed per sample:

w^(k) ← w^(k−1) − α ∇_w L(w^(k−1), ŷ^(i), y^(i))

where i = (k − 1) mod N

---

We consider a single neuron with sigmoid activation:

ŷ = σ(w^T x) = 1 / (1 + e^{−w^T x})

Binary cross-entropy loss with L2 regularization:

L(w, ŷ^(i), y^(i)) =
− [ y^(i) log(ŷ^(i)) + (1 − y^(i)) log(1 − ŷ^(i)) ] + λ ||w||^2

---

### 2.1.1
Download `data.csv` and generate a surface plot of J(w).

Compare:
- λ = 0
- λ = 10^{-2}

Hint: logarithmic scale

---

### 2.1.2
Derive the gradient:

∇_w L(w, ŷ^(i), y^(i))

Hint: σ'(z) = σ(z)(1 − σ(z))

---

### 2.1.3
Implement **BGD**:

- Fit model to `data.csv`
- Plot J(w) over iterations k = 1,...,K
- Plot trajectory of w^(k)

Configuration:
- K = 1000
- α = 10^{-1}
- λ = 10^{-2}
- w^(0) = [4, −7]^T

---

### 2.1.4
Repeat with **SGD**

- Compare BGD vs SGD
- Compute number of epochs
- Which method is better for large datasets?

---

### 2.1.5
Use JAX for automatic differentiation:

- Replace manual gradients with `jax.grad`
- Compare results with BGD

---

## Problem 2.2 (25%)

SVC objective:

x^T w + b ≥ 1 if y = 1  
x^T w + b ≤ −1 if y = −1  

Loss:

J(w, b) = (1/N) ∑ max(0, 1 − y^(i)(x^T w + b)) + λ||w||^2

Kernel trick:

ϕ: ℝ^d → ℝ^{d'}

---

### 2.2.1
Visualize hinge loss:

L = max(0, 1 − yŷ)

- Compare y = ±1
- Compare with Perceptron

---

### 2.2.2
Dataset: `blobs.csv`

- Plot data
- Train SVC (linear, C = 1)
- Extract w and b

---

### 2.2.3
- Plot decision boundary
- Plot margins: x^T w + b = ±1
- Compute margin width

---

### 2.2.4
Repeat with `circles.csv`

- Is it linearly separable?

---

### 2.2.5
Find transformation:

ϕ: ℝ^2 → ℝ^3  
ϕ(x) = [x0, x1, x2']^T

- Train SVC (C = 100)
- Plot transformed data
- Plot hyperplane (set x1 = 0)

---

### 2.2.6
Polynomial kernel (degree 2):

k(x_i, x_j) = (x_i^T x_j + 1)^2

- Train SVC
- Plot decision regions
- Is boundary linear in original space?

---

### 2.2.7
Kernel definition:

k(x_i, x_j) = ⟨ϕ(x_i), ϕ(x_j)⟩

Given:

k(x_i, x_j) = (x_i^T x_j + 1)^2

- Find explicit ϕ(x)
- What is dimension d'?

---

## Problem 2.3 (25%)

Ridge Regression:

J(w) = (1/2)||Xw − y||^2 + (1/2)λ||w||^2

Solution:

ŵ = (X^T X + λI)^{-1} X^T y

Prediction:

ŷ = X' ŵ

---

### 2.3.1
Show:

(X^T X + λI)^{-1} X^T = X^T (X X^T + λI)^{-1}

---

### 2.3.2
Derive kernel form:

ŷ = k(x')^T (K + λI)^{-1} y

Implement for polynomial kernel:

k(x_i, x_j) = (x_i^T x_j + 1)^m

---

### 2.3.3
Fit model (m = 1, 2, 3, 10)

- Plot predictions
- Analyze regularization

---

### 2.3.4
RBF kernel:

k(x_i, x_j) = exp(−||x_i − x_j||^2 / (2 l_s^2))

- Discuss role of l_s

---

### 2.3.5
Use Taylor expansion:

e^x = ∑ x^n / n!

- Derive feature mapping for RBF (d = 1)
- What is d'?

---

## Problem 2.4 (25%)

Goal: Probability calibration

---

### 2.4.1
Dataset: `calibration.csv`

- Split:
  - T_train_full (80%)
  - T_test (20%)
- Then:
  - T_train (80%)
  - T_cal (20%)

Use `random_state = 256`

---

### 2.4.2
Train Random Forest:

- Accuracy on T_test
- Confusion matrix
- Precision, Recall, F1

---

### 2.4.3
Reliability diagram (manual):

- Predict probabilities
- Bin with np.linspace(0,1,11)
- Compute:
  - x = mean predicted prob
  - y = fraction of positives

- Plot calibration curve
- Compute log-loss

---

### 2.4.4
Use:

CalibrationDisplay.from_estimator()

- Compare with manual version

---

### 2.4.5
Platt scaling:

P(y=1 | f) = 1 / (1 + exp(af + b))

- Use CalibratedClassifierCV(cv="prefit", method="sigmoid")
- Fit on T_cal
- Evaluate on T_test:
  - Accuracy
  - Log-loss
  - Reliability diagram

---

Optional:
- Try isotonic regression