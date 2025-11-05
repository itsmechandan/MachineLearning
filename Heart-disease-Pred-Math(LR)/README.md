# Heart Disease Prediction using Logistic Regression (Mathematical Implementation)

## 📘 Abstract
This project presents a detailed exploration of **Logistic Regression** applied to predict heart disease presence using a mathematical and programmatic approach. Unlike conventional implementations using high-level libraries, this project derives the **logistic hypothesis, cost function, and gradient descent updates** from first principles. The objective is to understand the underlying optimization mechanics that govern classification algorithms, with heart disease prediction serving as a practical case study.

---

## 🧩 1. Introduction
Cardiovascular diseases remain one of the leading causes of mortality worldwide. Early detection using computational methods can significantly reduce risk and improve clinical outcomes. This study applies **binary classification through Logistic Regression**, aiming to predict whether a patient has heart disease (1) or not (0) based on input health metrics.

The project is divided into two major phases:
1. **Mathematical Derivation & Manual Implementation** — implementing sigmoid, cost, and gradient descent functions from scratch.
2. **Model Validation using Scikit-learn** — comparing results with `LogisticRegression` from `sklearn` for verification and benchmarking.

---

## 🧮 2. Mathematical Background

### 2.1 Hypothesis Representation
The logistic model predicts probabilities as:

\[
f_{w,b}(x) = g(w \cdot x + b)
\]
where \( g(z) \) is the sigmoid function:
\[
g(z) = \frac{1}{1 + e^{-z}}
\]

### 2.2 Cost Function
For a dataset with \( m \) examples:
\[
J(w,b) = \frac{1}{m}\sum_{i=1}^{m} \left[ -y^{(i)}\log(f_{w,b}(x^{(i)})) - (1-y^{(i)})\log(1-f_{w,b}(x^{(i)})) \right]
\]

### 2.3 Gradient Descent Updates
The weights are updated iteratively to minimize cost:
\[
w_j := w_j - \alpha \frac{\partial J(w,b)}{\partial w_j}
\]
\[
b := b - \alpha \frac{\partial J(w,b)}{\partial b}
\]

---

## ⚙️ 3. Implementation Workflow

### Phase 1: Manual Mathematical Implementation
- Implemented the **sigmoid**, **cost function**, and **gradient descent** manually.
- Verified convergence by plotting cost over iterations.
- Predicted binary outcomes based on thresholded sigmoid probabilities.

### Phase 2: Library Validation (Scikit-learn)
- Reimplemented the same using:
  ```python
  from sklearn.linear_model import LogisticRegression
