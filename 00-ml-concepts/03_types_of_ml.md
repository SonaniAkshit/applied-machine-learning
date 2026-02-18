# Types of Machine Learning

```
Machine Learning
│
├── 1. Supervised Learning
│   │
│   ├── Regression (Continuous Output)
│   │   ├── Linear Regression
│   │   ├── Ridge Regression
│   │   ├── Lasso Regression
│   │   ├── Elastic Net
│   │   ├── Decision Tree Regressor
│   │   ├── Random Forest Regressor
│   │   ├── Gradient Boosting Regressor
│   │   └── Support Vector Regressor (SVR)
│   │
│   └── Classification (Categorical Output)
│       ├── Logistic Regression
│       ├── K-Nearest Neighbors (KNN)
│       ├── Decision Tree Classifier
│       ├── Random Forest Classifier
│       ├── Support Vector Machine (SVM)
│       ├── Naive Bayes
│       ├── Gradient Boosting (XGBoost / LightGBM / CatBoost)
│       └── Neural Networks (MLP)
│
├── 2. Unsupervised Learning
│   │
│   ├── Clustering
│   │   ├── K-Means
│   │   ├── Hierarchical Clustering
│   │   ├── DBSCAN
│   │   ├── Mean Shift
│   │   └── Gaussian Mixture Models (GMM)
│   │
│   ├── Dimensionality Reduction
│   │   ├── PCA
│   │   ├── LDA
│   │   ├── t-SNE
│   │   ├── UMAP
│   │   └── Autoencoders
│   │
│   ├── Anomaly Detection
│   │   ├── Isolation Forest
│   │   ├── One-Class SVM
│   │   ├── Local Outlier Factor (LOF)
│   │   └── Autoencoder-based Detection
│   │
│   └── Association Rule Learning
│       ├── Apriori
│       ├── FP-Growth
│       └── Eclat
│
├── 3. Semi-Supervised Learning
│   │
│   ├── Self-Training
│   ├── Label Propagation
│   ├── Co-Training
│   └── Semi-Supervised SVM
│
└── 4. Reinforcement Learning
    │
    ├── Model-Free
    │   ├── Q-Learning
    │   ├── SARSA
    │   ├── Deep Q Network (DQN)
    │   └── Policy Gradient
    │
    └── Model-Based
        ├── Monte Carlo Tree Search
        └── Dynamic Programming
```

---
# Supervised Learning

## Definition

Supervised Learning is a type of Machine Learning where the model learns a mapping function:

$$
f: X \rightarrow Y
$$

from labeled data.

* **X** = Input features (independent variables)
* **Y** = Output label (dependent variable)

The goal is to learn a function $f(x)$ that approximates the true relationship between input and output by minimizing a loss function.

---

## Core Idea

We have a dataset:

$$
D = {(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)}
$$

Each input $x_i$ has a corresponding correct output $y_i$.

The model learns parameters $\theta$ such that:

$$
\hat{y} = f(x; \theta)
$$

and minimizes:

$$
Loss = L(y, \hat{y})
$$

---

## Real-Life Example

### Student Placement Prediction

Features (X):

* IQ
* CGPA

Output (Y):

* Placement (Yes / No)

If we train a model on historical student data, it learns patterns such as:

* Higher CGPA + good IQ → Higher probability of placement
* Lower CGPA → Lower probability

After training, for a new student:

IQ = 115
CGPA = 8.2

Model predicts: **Yes**

---

## Visual Representation

![Supervised Learning Diagram](https://images.prismic.io/superpupertest/f3517801-371b-4be8-95f0-f18b3b7804f2_How-does-supervised-learning-work.webp)

![Input Output Mapping](https://www.researchgate.net/publication/366602691/figure/fig3/AS%3A11431281109559046%401672105877509/Schematic-diagram-of-input-and-output-variables-a-in-three-machine-learning-models.png)

![Image](https://cdn.labellerr.com/training%20data/Essential%20guide/training-data.webp)

---

# Types of Supervised Learning

Supervised learning has two main categories:

---

# 1. Regression

## Definition

Regression is used when the output variable is **continuous (numerical)**.

$$
Y \in \mathbb{R}
$$

Examples:

* Salary prediction
* House price prediction
* Temperature forecasting

---

## Real-Life Example

| IQ  | CGPA | Salary (LPA) |
| --- | ---- | ------------ |
| 80  | 8.0  | 7            |
| 95  | 8.5  | 9            |
| 110 | 9.0  | 12           |

If a new student has:

IQ = 100
CGPA = 8.3

Model predicts:

Salary ≈ 9.5 LPA

---

## Mathematical Objective

### Mean Squared Error (MSE)

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
$$

The model minimizes the squared difference between actual and predicted values.

---

## Visual Concept

![Regression Line](https://www.investopedia.com/thmb/U0vCJoyO7HDkx4lOsWeTYwxniDg%3D/1500x0/filters%3Ano_upscale%28%29%3Amax_bytes%28150000%29%3Astrip_icc%28%29/line-of-best-fit.asp-final-ed50f47f6cf34662846b3b89bf13ceda.jpg)

![Image](https://bookdown.org/a_shaker/STM1001_Topic_8/STM1001_Topic_8_files/figure-html/unnamed-chunk-14-1.svg)


![MSE Visualization](https://cdn-media-1.freecodecamp.org/images/MNskFmGPKuQfMLdmpkT-X7-8w2cJXulP3683)

---

# 2. Classification

## Definition

Classification is used when the output variable is **categorical (discrete classes)**.

Binary case:

$$
Y \in {0,1}
$$

Multi-class case:

$$
Y \in {Class_1, Class_2, ..., Class_k}
$$

Examples:

* Placement (Yes/No)
* Email (Spam / Not Spam)
* Disease (Positive / Negative)

---

## Real-Life Example

| IQ  | CGPA | Placement |
| --- | ---- | --------- |
| 85  | 7.5  | No        |
| 105 | 8.8  | Yes       |

For a new student:

IQ = 98
CGPA = 8.1

Model predicts probability:

$$
P(Placement = Yes) = 0.76
$$

Since probability > 0.5 → Output = Yes

---

## Mathematical Objective

### Binary Cross-Entropy Loss

$$
Loss = - \left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right]
$$

Instead of predicting direct labels, the model predicts probabilities.

---

## Classification Visual Concept

![Image](https://www.researchgate.net/publication/345986400/figure/fig1/AS%3A1000379500216321%401615520467602/The-linear-decision-boundary-for-binary-two-dimensional-banana-shaped-data-The-plot.png)


![Decision Boundary](https://scipython.com/media/old_blog/logistic_regression/decision-boundary.png)

![Linear Classifier Example](https://www.researchgate.net/profile/Cheng-Soon-Ong/publication/23442384/figure/fig1/AS%3A310235752353793%401450977375273/A-linear-classifier-separating-two-classes-of-points-squares-and-circles-in-two_Q320.jpg)

![Image](https://www.researchgate.net/publication/359803757/figure/fig1/AS%3A1147532809900067%401650604550888/llustration-of-linear-SVM-Classifier-separating-the-two-classes-llustration-of-linear_Q320.jpg)

---

## Decision Boundary Explanation

Model learns:

$$
f(x) = w^T x + b
$$

If:

$$
f(x) > 0 \Rightarrow Class\ 1
$$

Else:

$$
Class\ 0
$$

---

## Logistic Regression Probability Function

Sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Output range: (0,1)

If probability > 0.5 → Class 1
Else → Class 0

---

# Key Differences: Regression vs Classification

| Aspect       | Regression   | Classification |
| ------------ | ------------ | -------------- |
| Output Type  | Continuous   | Discrete       |
| Example      | Salary       | Yes/No         |
| Loss         | MSE          | Cross-Entropy  |
| Output Space | Real numbers | Finite classes |

---

# Important Concepts in Supervised Learning

* Labeled Data Required
* Feature Engineering
* Train-Test Split
* Overfitting & Underfitting
* Bias-Variance Tradeoff
* Evaluation Metrics (RMSE, Accuracy, F1-score, etc.)

---