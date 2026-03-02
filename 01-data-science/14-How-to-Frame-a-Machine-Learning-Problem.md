# How to Frame a Machine Learning Problem

---

## 1. Intuition

Machine learning does not start with a model.

It starts with a **business pain**.

If you don’t clearly define:

* what problem you are solving
* why it matters
* how success will be measured

then even a perfect model is useless.

Framing means:

> Convert a vague business goal into a precise ML objective with clear inputs, outputs, and metrics.

---

## 2. From Business Problem → ML Problem

### Example: Netflix Revenue Growth

Business problem:

> How do we increase revenue?

Possible strategies:

1. Acquire new customers
2. Increase subscription price
3. Reduce churn

Now think like an ML engineer.

### Evaluate Risk of Each Strategy

| Strategy              | Risk                  | Cost      | Control      |
| --------------------- | --------------------- | --------- | ------------ |
| Acquire new customers | High marketing cost   | Very high | Hard         |
| Increase price        | Customers may leave   | Medium    | Risky        |
| Reduce churn          | Retain existing users | Lower     | High control |

Reducing churn is usually the smartest ML-driven strategy.

Why?

Because:

* You already have data.
* Retention is cheaper than acquisition.
* You can act early.

So business decision:

> Focus on churn reduction.

---

## 3. Converting to ML Problem

Now you must define:

### What exactly is churn?

Example definition:

> A user who cancels subscription within next 30 days.

Now we translate:

Input:

* User watch time
* Search history
* Click behavior
* Subscription plan
* Payment history

Output:

* Will the user churn? (Yes/No)

This becomes:

> Binary Classification Problem

If output was:

* Probability of churn → still classification
* Number of days until churn → regression
* Group similar users → clustering

So model type depends on **target variable**.

---

## 4. Define the Target Variable Clearly

This is where beginners fail.

You must answer:

* What is churn?
* Over what time window?
* Is it immediate churn or future churn?

Example:

Target = 1
If user cancels within next 30 days

Target = 0
If user stays active

Now the problem is mathematically defined.

---

## 5. Check Existing Solutions (Current System)

Before building anything:

Ask:

* Is there already a churn dashboard?
* Is business already calculating churn rate?
* Is there rule-based logic?

Industry rule:

> Never build from scratch if something already works.

Sometimes:

* There is already a logistic regression model.
* You just need to improve features.

This saves months.

---

## 6. Data Collection Strategy

You mentioned good points. Let’s structure them.

### Possible Features for Churn Model

* Watch time per week
* Drop-off in watch time
* Search but did not find content
* Incomplete watching
* Click-through rate on recommendations
* Plan type
* Payment failures
* Customer complaints

Important question:
Is this data already stored?

If not:

* Can we track it?
* Is tracking legal?
* Is tracking scalable?

Data availability decides feasibility.

---

## 7. Choose the Type of ML Problem

Now decide formally:

### If predicting churn (Yes/No)

Binary Classification

### If predicting churn probability

Binary Classification with probability output

### If predicting number of days left

Regression

### If grouping risky users

Clustering (unsupervised)

In most real churn systems:

> Start with binary classification.

---

## 8. Define Evaluation Metrics

Do NOT jump to accuracy.

For churn:

If dataset is imbalanced (usually it is):

Example:
90% users stay
10% churn

Accuracy can be misleading.

Better metrics:

* Precision
* Recall
* F1 Score
* ROC-AUC

Business aligned metric:

> Recall for churn class

Why?

Because missing a churner is expensive.

---

## 9. Online vs Batch Prediction

Important production decision.

### Batch

* Run once daily
* Predict churn risk for all users
* Send retention emails

Use when:

* Real-time not required

### Online (Real-time)

* Predict immediately when user activity drops
* Trigger instant discount offer

Use when:

* Immediate intervention increases retention

Batch is simpler.
Online is harder but more powerful.

Start simple unless business demands real-time.

---

## 10. Check Assumptions

Every ML model assumes:

* Future behavior similar to past
* Data is clean
* Labels are correct
* No data leakage

Example leakage:
Using “subscription cancelled date” as feature while predicting churn.

That would make model perfect but useless.

---

## 11. Mathematical Framing

For binary classification:

We model:

[
P(Y=1 \mid X)
]

Where:

* ( Y = 1 ) → user churns
* ( X ) → user features

Loss function commonly:

[
L = - \frac{1}{n} \sum_{i=1}^{n} [ y_i \log(p_i) + (1 - y_i)\log(1 - p_i) ]
]

This is Binary Cross Entropy.

This connects business to math.

---

## 12. Production-Level Thinking

Ask these questions:

* How often will model retrain?
* What if user behavior changes?
* How will we monitor drift?
* How will business act on predictions?
* What is cost of false positive?

Important:

Model alone does nothing.

You need:

Prediction → Decision → Action → Measurement

Without action pipeline, model is useless.

---

## 13. Full Framing Checklist (Industry Standard)

Before building:

1. What business metric are we improving?
2. How exactly is target defined?
3. What is prediction horizon?
4. What type of ML problem?
5. What is baseline?
6. What metric matters?
7. What is acceptable error?
8. Is data available and reliable?
9. Batch or real-time?
10. How will model output be used?

If you cannot answer these clearly:
You are not ready to build.

---

## 14. Common Beginner Mistakes

* Jumping to Random Forest first
* Not defining churn window
* Ignoring class imbalance
* Using data leakage
* Not thinking about deployment
* Optimizing accuracy instead of business value

---

## Final Mental Model

Business Problem
↓
Choose Best Strategy
↓
Define Target Clearly
↓
Translate to ML Type
↓
Check Existing Solution
↓
Collect Data
↓
Define Metrics
↓
Decide Batch vs Online
↓
Check Assumptions
↓
Then build model

---