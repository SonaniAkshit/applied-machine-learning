
# **“AI vs ML vs DL”**

**Structure:**
```
AI (umbrella)
└── ML (subset)
  └── DL (subset of ML)
```
---

# What is Artificial Intelligence?

Artificial Intelligence (AI) is a broad field of computer science focused on building systems that can perform tasks which normally require human intelligence.

These tasks include:

* Making decisions
* Solving problems
* Understanding language
* Recognizing images
* Learning from experience

AI does not mean machines think like humans.
It means machines are designed to behave intelligently in specific tasks.

---

## Simple Way to Understand AI

If a machine can:

* Take input
* Process it
* Make a decision
* Act based on that decision

in a way that appears intelligent, it is considered AI.

---

## Important Clarification

AI is a broad umbrella term.

* Some AI systems use fixed rules (rule-based systems).
* Some AI systems learn from data (Machine Learning).
* Some use deep neural networks (Deep Learning).

Not all AI systems learn.
Learning-based AI is called Machine Learning.

---

# What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence that enables systems to learn patterns from data and improve their performance without being explicitly programmed for every scenario.

Instead of writing fixed rules, we provide data to the system. The model analyzes this data, identifies patterns, and uses those patterns to make predictions or decisions on new, unseen data.

---

## Key Characteristics of Machine Learning

* Learns from historical data
* Improves performance over time
* Makes predictions or classifications
* Reduces the need for manual rule-writing

---

## Simple Example

In spam email detection:

* Instead of writing fixed rules for every spam word,
* A Machine Learning model is trained on thousands of labeled emails.
* The model learns patterns and predicts whether a new email is spam or not.

---

## Difference clarity:

- AI → intelligent behavior

- ML → learning mechanism behind that behavior

---

# Deep Learning in the Context of Machine Learning

## 1. What is Deep Learning?

Deep Learning (DL) is a subset of Machine Learning that uses Artificial Neural Networks with multiple layers to automatically learn patterns from data.

In traditional Machine Learning, humans design features manually.
In Deep Learning, the model learns features automatically from raw data.

Deep Learning is especially powerful when dealing with complex data such as:

* Images
* Audio
* Text
* Video

---

## 2. How Deep Learning is Different from Traditional ML

### Traditional Machine Learning

* Human designs features manually.
* Model learns relationship between features and output.
* Works well for structured/tabular data.

Example:
For image classification:

* Extract edges
* Detect shapes
* Measure color intensity
  Then apply a classifier.

Human → Feature Engineering
Model → Prediction

---

### Deep Learning

* Raw data is given directly to the model.
* The model automatically learns features at multiple levels.
* No manual feature engineering required (in most cases).

Model → Feature Learning + Prediction

This automatic feature learning is the key strength of Deep Learning.

---

## 3. Why is it Called “Deep”?

The word “Deep” refers to multiple layers in a neural network.

A Deep Neural Network typically has:

* Input Layer
* Multiple Hidden Layers
* Output Layer

More hidden layers = deeper network.

Each layer learns a more abstract representation of the data.

Example (Image Recognition):

* Layer 1: Detect edges
* Layer 2: Detect shapes
* Layer 3: Detect parts (eyes, wheels)
* Layer 4: Detect full objects

The depth allows the system to learn hierarchical patterns.

---

## 4. Artificial Neurons (Core Concept)

The basic unit of a neural network is called an **Artificial Neuron**.

It is inspired by biological neurons in the human brain.

### Structure of an Artificial Neuron

An artificial neuron:

1. Takes multiple inputs
2. Multiplies each input with a weight
3. Adds a bias
4. Passes the result through an activation function
5. Produces an output

Mathematically:

Output = Activation(Weighted Sum of Inputs + Bias)

---

## 5. Components of an Artificial Neuron

### 1. Inputs

Features or values given to the model.

### 2. Weights

Each input has a weight.
Weights represent importance.

Higher weight → More influence on output.

### 3. Bias

Helps shift the output.
Improves flexibility of the model.

### 4. Activation Function

Adds non-linearity.
Without activation, deep networks cannot learn complex patterns.

Common activation functions:

* ReLU
* Sigmoid
* Tanh

---

## 6. Neural Networks

When multiple artificial neurons are connected together in layers, it forms a Neural Network.

Input Layer → Hidden Layers → Output Layer

Each neuron in one layer connects to neurons in the next layer.

The network learns by adjusting weights using a process called:

**Backpropagation**

This process minimizes error between predicted output and actual output.

---

## 7. When to Use Deep Learning

Deep Learning is useful when:

* Large amount of data is available
* Data is unstructured (image, text, speech)
* Problem is complex
* High accuracy is required

For small tabular datasets, traditional ML often performs better and is simpler.

---

## 8. Summary

* Deep Learning is a subset of Machine Learning.
* It uses Artificial Neural Networks with multiple layers.
* It automatically learns features from raw data.
* It performs exceptionally well on complex, high-dimensional data.
* It requires large data and high computational power.

---