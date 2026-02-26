# Important Mathematical Symbols in Machine Learning

## 1️⃣ What is `θ`?

**Symbol:** $\theta$
**Pronunciation:** *theta*

### Meaning in Machine Learning

* Represents **model parameters**
* Includes weights and bias
* These are the values the model learns during training

### Example

$$
\hat{y} = f(x; \theta)
$$

Here:

* $\theta$ = learnable parameters
* $\hat{y}$ = predicted output
* $x$ = input

---

# Important Symbols Table

| Symbol       | Pronunciation | Meaning in ML        | Example                        |
| ------------ | ------------- | -------------------- | ------------------------------ |
| $X$          | X             | Input feature matrix | Training dataset               |
| $x_i$        | x sub i       | ith training example | Single data point              |
| $y$          | y             | True output          | Actual label                   |
| $\hat{y}$    | y hat         | Predicted output     | Model prediction               |
| $\theta$     | theta         | Model parameters     | Weights                        |
| $w$          | w             | Weight vector        | Linear model weight            |
| $b$          | b             | Bias term            | Intercept                      |
| $L$          | L             | Loss function        | Error measure                  |
| $J(\theta)$  | J of theta    | Cost function        | Optimization objective         |
| $\alpha$     | alpha         | Learning rate        | Step size in GD                |
| $\nabla$     | nabla         | Gradient operator    | Direction of steepest increase |
| $\sigma$     | sigma         | Sigmoid function     | Logistic regression            |
| $\mu$        | mu            | Mean                 | Gaussian mean                  |
| $\Sigma$     | capital sigma | Covariance matrix    | Multivariate Gaussian          |
| $\sum$       | summation     | Add all terms        | MSE                            |
| $\prod$      | product       | Multiply terms       | Likelihood                     |
| $\mathbb{R}$ | R             | Real numbers         | Regression output              |
| $\in$        | belongs to    | Set membership       | $y \in {0,1}$                  |
| $\partial$   | partial       | Partial derivative   | Backpropagation                |
| $\log$       | log           | Logarithm            | Cross-entropy                  |
| $e$          | e             | Euler’s number       | Sigmoid                        |

---

# Important Concept Clarifications

## 1️⃣ Difference Between $L(y, \hat{y})$ and $J(\theta)$

### Loss Function

$$
L(y, \hat{y})
$$

* Measures error for **one training example**
* Example (for regression):

$$
L(y, \hat{y}) = (y - \hat{y})^2
$$

This tells how wrong the prediction is for one data point.

---

### Cost Function

$$
J(\theta)
$$

* Measures total error over the **entire dataset**
* It depends on model parameters $\theta$

Example:

$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{y_i})
$$

So:

* $L$ → single example error
* $J$ → average error across dataset

Important:

We minimize $J(\theta)$, not just $L$.

---

## 2️⃣ Why Do We Take Derivative w.r.t. $\theta$ and Not w.r.t. $x$?

Model equation:

$$
\hat{y} = f(x; \theta)
$$

During training:

* $x$ is fixed (data is given)
* $\theta$ is variable (parameters we want to optimize)

We want to find optimal $\theta$ such that:

$$
\theta^* = \arg\min_\theta J(\theta)
$$

So we compute gradient:

$$
\nabla_\theta J(\theta)
$$

Why not derivative w.r.t $x$?

Because:

* $x$ is input data
* We do NOT change data
* We only update parameters

If we took derivative w.r.t $x$, we would be changing input data, which is not the objective in supervised learning.

Training goal:

Adjust $\theta$, not $x$.

---

# Core Machine Learning Structure

Machine Learning fundamentally works on:

1. Function approximation
   $$
   f(x; \theta)
   $$

2. Loss minimization
   $$
   L(y, \hat{y})
   $$

3. Cost minimization
   $$
   J(\theta)
   $$

4. Optimization
   $$
   \theta := \theta - \alpha \nabla_\theta J(\theta)
   $$

5. Probability (for classification)
   $$
   \sigma(z) = \frac{1}{1 + e^{-z}}
   $$

---