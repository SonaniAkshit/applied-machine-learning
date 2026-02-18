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

## Key Differences: Regression vs Classification

| Aspect       | Regression   | Classification |
| ------------ | ------------ | -------------- |
| Output Type  | Continuous   | Discrete       |
| Example      | Salary       | Yes/No         |
| Loss         | MSE          | Cross-Entropy  |
| Output Space | Real numbers | Finite classes |

---

## Important Concepts in Supervised Learning

* Labeled Data Required
* Feature Engineering
* Train-Test Split
* Overfitting & Underfitting
* Bias-Variance Tradeoff
* Evaluation Metrics (RMSE, Accuracy, F1-score, etc.)

---

# Unsupervised Learning

## Classical Definition

Unsupervised Learning is a type of Machine Learning where the model learns patterns, structure, or relationships from **unlabeled data**.

Unlike supervised learning, there is **no target variable $Y$**.

We are given only:

$$
X = {x_1, x_2, ..., x_n}
$$

The goal is to discover hidden structure in data.

---

# Core Idea

Given dataset:

$$
D = {x_1, x_2, ..., x_n}
$$

Each data point:

$$
x_i \in \mathbb{R}^d
$$

There is **no label $y_i$**.

The model tries to:

* Group similar data points
* Reduce dimensions
* Detect rare patterns
* Discover associations

---

## Visual Overview

![Image](https://images.openai.com/static-rsc-3/OC3fWRRLarHMcP1VTQlhW0mUTZ4y80CiBJjT5LvPDX_gWnzYc_aPuikd08614zAq72_Nmkk2-VnbXYhs0HnsF8w43XEkXTBuoEoa1YePzd8?purpose=fullsize\&v=1)

![Image](https://www.researchgate.net/publication/344017242/figure/fig2/AS%3A930941392396290%401598965132245/Cluster-Visualization-for-the-2D-3D-k-Means-Algorithm.png)

![Image](https://substackcdn.com/image/fetch/%24s_%21_xRi%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa00f3301-9f4f-4de3-a5bb-dc8866c1afc4_819x580.png)

![Image](https://miro.medium.com/1%2AT7CqlFV5aRm6MxO5nJt7Qw.gif)

---

# Types of Unsupervised Learning

---

# 1️⃣ Clustering

## Definition

Clustering is the task of grouping similar data points together such that:

* Points within same cluster are similar
* Points in different clusters are dissimilar

---

## Mathematical Objective (K-Means Example)

Minimize within-cluster variance:

$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2
$$

Where:

* $K$ = number of clusters
* $C_k$ = cluster k
* $\mu_k$ = centroid of cluster k

Goal: minimize total squared distance from cluster center.

---
## Visual Concept


![Image](https://i.sstatic.net/VqdbM.png)

![Image](https://uc-r.github.io/public/images/analytics/clustering/hierarchical/unnamed-chunk-13-1.png)

![Image](https://www.researchgate.net/publication/377173830/figure/fig2/AS%3A11431281361502724%401744122148309/Visualization-of-DBSCAN-clustering-algorithm.tif)

![Image](https://www.sthda.com/sthda/RDoc/figure/clustering/dbscan-density-based-clustering-dbscan-factoextra-ggplot2-1.png)

## What It Shows

* Data points divided into multiple groups
* Each cluster has internal similarity
* Different clusters are separated
---
## Real-Life Example

Student dataset:

| IQ  | CGPA |
| --- | ---- |
| 85  | 7.5  |
| 110 | 9.0  |
| 70  | 6.5  |

Model may create:

* High performance cluster
* Medium performance cluster
* Low performance cluster

After clustering, you may assign business labels.

Important: Labels are assigned **after clustering**, not during training.

---

# 2️⃣ Dimensionality Reduction

## Definition

Dimensionality Reduction transforms high-dimensional data into lower-dimensional space while preserving maximum information.

If:

$$
x_i \in \mathbb{R}^{100}
$$

We transform into:

$$
z_i \in \mathbb{R}^{2}
$$

---

## Mathematical Objective (PCA)

Maximize variance:

$$
\max ; Var(w^T X)
$$

Subject to:

$$
||w|| = 1
$$

PCA finds direction of maximum variance.

---
## Visual Concept


![Image](https://www.researchgate.net/publication/345602552/figure/fig1/AS%3A1028064637104131%401622121118891/PCA-Example-3D-to-2D.png)

![Image](https://builtin.com/sites/www.builtin.com/files/inline-images/national/Principal%2520Component%2520Analysis%2520second%2520principal.gif)

![Image](https://www.mathworks.com/help/examples/stats/win64/VisualizeHighDimensionalDataUsingTSNEExample_01.png)

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/T-SNE_visualisation_of_word_embeddings_generated_using_19th_century_literature.png/1280px-T-SNE_visualisation_of_word_embeddings_generated_using_19th_century_literature.png)

## What It Shows

* High-dimensional data projected into 2D
* Maximum variance direction preserved
* Useful for visualization

---
## Real-Life Example

* Image dataset with 784 pixels
* Reduce to 50 principal components
* Faster computation
* Less noise

Not just for “model fit”, but for:

* Visualization
* Noise reduction
* Speed

---

# 3️⃣ Anomaly Detection

## Definition

Anomaly Detection identifies rare data points that significantly differ from majority distribution.

---

## Mathematical View (Density-Based)

Estimate probability density:

$$
p(x)
$$

If:

$$
p(x) < \epsilon
$$

Then point is anomaly.

---
## Visual Concept


![Image](https://www.researchgate.net/publication/342155553/figure/fig3/AS%3A1004399522492439%401616478915973/Scatter-plot-of-the-results-from-LOF.png)

![Image](https://www.researchgate.net/profile/Jason-Mckenna/publication/381710897/figure/fig1/AS%3A11431281255160847%401719409715744/solation-forest-analysis-visualization-The-isolation-forest-was-fitted-on-the-entire_Q320.jpg)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2A2whcqVti2YUeAw6SOaMTQQ.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/0%2AC5gZBpC57Pdbnnik)

## What It Shows

* Normal data clustered together
* One or few points far from distribution
* Those are anomalies

---
## Real-Life Example

Credit card transactions:

* Normal transactions cluster together
* Rare unusual transaction → anomaly

Used in:

* Fraud detection
* Network security
* Manufacturing defect detection

---

# 4️⃣ Association Rule Learning

## Definition

Association Rule Learning discovers relationships between variables in transactional data.

---

## Example (Market Basket Analysis)

If customers buy:

Milk → Eggs

We compute:

### Support

$$
Support(A) = \frac{\text{Number of transactions containing A}}{\text{Total transactions}}
$$

### Confidence

$$
Confidence(A \rightarrow B) = \frac{Support(A \cap B)}{Support(A)}
$$

### Lift

$$
Lift = \frac{Confidence(A \rightarrow B)}{Support(B)}
$$

If Lift > 1 → strong association.

---
## Visual Concept


![Image](https://ars.els-cdn.com/content/image/3-s2.0-B978012381479100006X-f06-01-9780123814791.jpg)

![Image](https://www.researchgate.net/publication/352111791/figure/fig4/AS%3A11431281122742143%401677516536278/Flow-chart-of-the-Apriori-algorithm.jpg)

![Image](https://www.researchgate.net/publication/337999958/figure/fig1/AS%3A867641866604545%401583873349676/Formulae-for-support-confidence-and-lift-for-the-association-rule-X-Y.ppm)

![Image](https://www.researchgate.net/publication/337999958/figure/fig1/AS%3A867641866604545%401583873349676/Formulae-for-support-confidence-and-lift-for-the-association-rule-X-Y_Q320.jpg)

## What It Shows

* Item relationships
* Frequent itemsets
* Strong rule connections

---

## Key Differences from Supervised Learning

| Aspect        | Supervised        | Unsupervised             |
| ------------- | ----------------- | ------------------------ |
| Labels        | Required          | Not required             |
| Goal          | Predict output    | Discover structure       |
| Loss Function | Defined w.r.t Y   | Often internal objective |
| Example       | Salary prediction | Customer segmentation    |

---

## Important Insight

Unsupervised learning does not "predict output".

It discovers:

* Structure
* Distribution
* Similarity
* Latent features

---

# Semi-Supervised Learning

## Classical Definition

Semi-Supervised Learning is a machine learning approach where the model is trained using:

* A small amount of labeled data
* A large amount of unlabeled data

Formally:

We have dataset:

$$
D = D_L \cup D_U
$$

Where:

$$
D_L = {(x_1, y_1), ..., (x_l, y_l)}
$$

Labeled dataset

and

$$
D_U = {x_{l+1}, ..., x_{l+u}}
$$

Unlabeled dataset

with:

$$
u \gg l
$$

Meaning: unlabeled data is much larger than labeled data.

---

# Core Idea

Instead of ignoring unlabeled data, the model uses structure in $D_U$ to improve learning.

The assumption:

> Data points close to each other are likely to share the same label.

This is called the **Cluster Assumption**.

---

# Visual Overview

![Image](https://images.prismic.io/superpupertest/f3517801-371b-4be8-95f0-f18b3b7804f2_How-does-supervised-learning-work.webp?auto=compress%2Cformat\&dpr=3)

![Image](https://miro.medium.com/1%2A25tiPu7Dfgg0D_sB-tDYbA.png)

![Image](https://www.researchgate.net/publication/348321139/figure/fig1/AS%3A977543393980417%401610075915777/A-toy-example-of-label-propagation-LP-on-graphs-where-the-dark-blue-and-red-nodes.png)

![Image](https://www.researchgate.net/publication/340627329/figure/fig2/AS%3A880226183704577%401586873684535/An-example-of-the-Label-Propagation-Algorithm.ppm)

---

# How Semi-Supervised Learning Works

## Step 1: Train on Labeled Data

Train model using:

$$
L(y, \hat{y})
$$

on small labeled dataset $D_L$.

---

## Step 2: Predict on Unlabeled Data

For each $x \in D_U$:

Model computes:

$$
\hat{y} = f(x; \theta)
$$

If prediction confidence is high:

$$
P(\hat{y} \mid x) > \tau
$$

Then assign pseudo-label:

$$
(x, \hat{y})
$$

---

## Step 3: Retrain Model

Combine:

$$
D_L \cup \text{Pseudo-Labeled Data}
$$

Retrain model for better generalization.

---

# Real-Life Example

## Google Photos Face Recognition

* You label few images as "Papa"
* System learns facial embedding pattern
* It finds similar faces in unlabeled photos
* High similarity → auto-labels as "Papa"

Behind the scenes:

* Feature embedding network
* Similarity clustering
* Confidence threshold

---

## Common Algorithms

* Self-Training
* Label Propagation
* Semi-Supervised SVM
* Consistency Regularization (used in deep learning)

---

## Mathematical Objective

Total loss becomes combination of supervised and unsupervised parts:

$$
J(\theta) =
\underbrace{L_{supervised}}*{\text{on } D_L}
+
\lambda
\underbrace{L*{unsupervised}}_{\text{on } D_U}
$$

Where:

* $\lambda$ controls influence of unlabeled data
* $L_{unsupervised}$ often enforces consistency or smoothness

---

## Why Use Semi-Supervised Learning?

Labeling data is expensive.

Examples:

* Medical imaging
* Speech recognition
* Face recognition
* Industrial inspection

But raw data is abundant.

Semi-supervised learning reduces labeling cost while improving accuracy.

---

## Key Assumptions

1. Cluster assumption
2. Manifold assumption
3. Smoothness assumption

---

# Reinforcement Learning

## Classical Definition

Reinforcement Learning (RL) is a learning paradigm where an **agent interacts with an environment** and learns a policy to maximize cumulative reward over time.

Unlike supervised learning:

* No labeled dataset
* Feedback comes as reward signal
* Reward may be delayed

---

# Core Components

Reinforcement Learning is defined by a tuple:

$$
(S, A, R, P, \gamma)
$$

Where:

* $S$ = Set of states
* $A$ = Set of actions
* $R$ = Reward function
* $P$ = State transition probability
* $\gamma$ = Discount factor

---

# Interaction Loop

At time step $t$:

1. Agent observes state $s_t$
2. Takes action $a_t$
3. Receives reward $r_t$
4. Environment transitions to new state $s_{t+1}$

Goal:

$$
\max_\pi ; \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

Where:

* $\pi$ = policy
* $\gamma \in (0,1)$ controls importance of future reward

---

# Visual Overview

![Image](https://images.openai.com/static-rsc-3/pVJ-grkB5Abhp4Yp_qz-5M99qf2vjONvXM2C0T0iQAAn45VXncj9ggnPAUHYDdicQWekqlhsSLgg7whqrmezS242J8p00aF55Tj-QH3RkY8?purpose=fullsize\&v=1)

![Image](https://images.deepai.org/glossary-terms/f387c0a8f57547a4a68c9afcbbc11494/main-qimg-f92c275af47e561651857f9af6bb85e9.png)

![Image](https://api.wandb.ai/files/cosmo3769/images/projects/38225538/7ac907b1.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2ACjpsqXkwkfkrzxZYR3smMg.png)

---

# Key Concepts

## 1️⃣ Policy ($\pi$)

Policy defines action selection rule:

$$
a = \pi(s)
$$

Can be:

* Deterministic
* Stochastic

---

## 2️⃣ Value Function

Expected cumulative reward from state $s$:

$$
V^\pi(s) = \mathbb{E}*\pi \left[ \sum*{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right]
$$

---

## 3️⃣ Q-Function

Expected reward for taking action $a$ in state $s$:

$$
Q^\pi(s,a)
$$

---

# Example: Game Playing

Agent = Player
Environment = Game

* Move left → reward 0
* Collect coin → reward +1
* Fall into trap → reward -5

Model learns which sequence of actions gives highest total reward.

---

## Real-Life Applications

* Robotics
* Self-driving cars
* Game AI (Chess, Go)
* Recommendation systems
* Trading systems

---

## Why It Is Different from Supervised Learning

| Aspect    | Supervised              | Reinforcement            |
| --------- | ----------------------- | ------------------------ |
| Feedback  | Immediate correct label | Delayed reward           |
| Data      | Fixed dataset           | Generated by interaction |
| Objective | Minimize loss           | Maximize reward          |

---

## Important Insight

Reinforcement Learning optimizes:

Long-term reward, not immediate correctness.

---

# 1️⃣ Supervised Learning – Real-World Applications

## 🔹 Where It Dominates

Used when labeled historical data is available.

---

## A. Finance & FinTech

### Regression

* Credit risk scoring
* Loan default probability prediction
* Stock price forecasting
* Insurance premium estimation

### Classification

* Fraud detection
* Credit approval (approve / reject)
* Customer churn prediction
* Anti-money laundering flagging

---

## B. Healthcare

### Classification

* Disease diagnosis (cancer detection)
* Medical image classification
* ICU mortality prediction

### Regression

* Hospital stay duration prediction
* Drug dosage prediction
* Disease progression modeling

---

## C. E-commerce

### Classification

* Spam review detection
* Product return prediction
* Customer segmentation (supervised segmentation)

### Regression

* Demand forecasting
* Revenue prediction
* Dynamic pricing

---

## D. Computer Vision

* Image classification
* Face recognition
* Object detection
* Autonomous driving perception systems

Dominant type here:
Deep supervised learning.

---

## E. NLP (Natural Language Processing)

* Sentiment analysis
* Text classification
* Named entity recognition
* Machine translation

---

# 2️⃣ Unsupervised Learning – Real-World Applications

## 🔹 Where It Dominates

When labels are unavailable or unknown.

---

## A. Customer Segmentation

Type: Clustering

* Retail segmentation
* Telecom user grouping
* Marketing persona generation

Domain: Marketing Analytics

---

## B. Recommendation Systems (Preprocessing)

Type: Dimensionality Reduction + Clustering

* Embedding generation
* User behavior grouping
* Collaborative filtering preprocessing

Domain: Big Tech (Amazon, Netflix, YouTube)

---

## C. Fraud & Security

Type: Anomaly Detection

* Credit card fraud detection
* Intrusion detection systems
* Network traffic monitoring

Domain: Cybersecurity

---

## D. Manufacturing

Type: Anomaly Detection

* Defect detection
* Equipment failure prediction
* Sensor data monitoring

Domain: Industrial IoT

---

## E. Retail Analytics

Type: Association Rule Learning

* Market basket analysis
* Cross-selling strategy
* Product bundling optimization

Domain: Retail chains, supermarkets

---

# 3️⃣ Semi-Supervised Learning – Real-World Applications

## 🔹 Where Labeling is Expensive

---

## A. Medical Imaging

* MRI tumor detection
* Rare disease classification
* Pathology slide analysis

Reason:
Labeled data is scarce and costly.

---

## B. Speech Recognition

* Voice assistants
* Accent adaptation
* Language modeling

Companies:
Google, Amazon, Apple

---

## C. Face Recognition

* Social media tagging
* Surveillance systems
* Identity verification

---

## D. Autonomous Driving

* Object detection with limited labeled frames
* Scene segmentation

---

## E. NLP Large Models

Modern LLM pretraining uses:

* Massive unlabeled data
* Fine-tuning on small labeled datasets

---

# 4️⃣ Reinforcement Learning – Real-World Applications

## 🔹 Where Sequential Decision Making Matters

---

## A. Robotics

* Robot arm control
* Warehouse automation
* Drone navigation

---

## B. Autonomous Vehicles

* Lane change decision
* Route optimization
* Adaptive cruise control

---

## C. Game AI

* Chess engines
* Go
* Strategy games

---

## D. Trading & Finance

* Portfolio optimization
* Algorithmic trading
* Market-making strategies

---

## E. Recommendation Systems

* Dynamic content ranking
* Ad placement optimization
* Personalized feeds

---

# Summary Table

| Domain             | Supervised | Unsupervised | Semi-Supervised | Reinforcement |
| ------------------ | ---------- | ------------ | --------------- | ------------- |
| Finance            | High       | Medium       | Low             | Medium        |
| Healthcare         | High       | Medium       | High            | Low           |
| Retail             | High       | High         | Low             | Low           |
| Cybersecurity      | Medium     | High         | Low             | Low           |
| Robotics           | Low        | Low          | Low             | High          |
| Autonomous Systems | Medium     | Low          | Medium          | High          |
| NLP                | High       | Medium       | High            | Medium        |

---

# Important Industry Insight

1. Supervised Learning dominates production systems.
2. Unsupervised is mostly used for preprocessing or discovery.
3. Semi-supervised is growing fast in medical and AI research.
4. Reinforcement Learning is powerful but expensive and complex to deploy.

---