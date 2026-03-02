# 🎯 Project: Placement Prediction (Toy but Structured)

Before jumping into deep ML, you build a **small, controlled, end-to-end project**. Just like CRUD in web dev.

This is your ML CRUD.

We’ll treat this seriously. Not as “just toy”.
Even toy projects must follow industry thinking.

## Problem Statement

Given:

* `cgpa`
* `iq`

Predict:

* `placement` → 1 (Placed) or 0 (Not Placed)

This is a **binary classification problem**.

Now we do this properly.

---

# Step -1: First Think (Before Touching Code)

Ask yourself:

* Is placement really determined by only CGPA and IQ?
* Is this linearly separable?
* Is data balanced?
* What assumptions are we making?

Assumption here:

> Higher CGPA + Higher IQ → Higher probability of placement

That’s a strong assumption.
In real world, this is incomplete. But fine for learning.

---

# Step 0: Create Synthetic Dataset (1000 rows)

We simulate realistic data.

### Idea:

* CGPA range: 4.0 – 9.5
* IQ range: 80 – 160
* Placement probability increases with both

We generate placement using a logical rule + noise.

### Code (Dataset Creation)

```python
import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

cgpa = np.round(np.random.uniform(4.0, 9.5, n), 2)
iq = np.random.randint(80, 161, n)

# Create probability based on weighted formula
prob = (0.6 * (cgpa / 10)) + (0.4 * (iq / 200))

placement = np.where(prob > 0.5, 1, 0)

df = pd.DataFrame({
    "cgpa": cgpa,
    "iq": iq,
    "placement": placement
})

df.to_csv("placement_data.csv", index=False)

print(df.head())
```

This creates your `.csv` file.

---

# Now Real ML Pipeline Begins

---

# Step 0: Preprocessing + EDA + Feature Selection

## Why?

You never trust raw data.

### Check:

```python
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### What you are checking:

* Missing values?
* Data types correct?
* Outliers?
* Class imbalance?

If placement has:

* 900 ones and 100 zeros → model biased.

---

# Step 1: Extract Input and Output

```python
X = df[['cgpa', 'iq']]
y = df['placement']
```

Simple. Clean.

---

# Step 2: Scaling

Why scale?

* IQ range ~ 80–160
* CGPA range ~ 4–9

Different scales → model may bias toward larger scale feature.

Use StandardScaler.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Important:

In production → never fit scaler again. Only transform.

---

# Step 3: Train Test Split

Why?

Model should generalize to unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

80% training
20% testing

---

# Step 4: Train Model

Start simple.

Logistic Regression.

Why?

* Binary classification
* Linear boundary
* Interpretable

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

# Step 5: Evaluate Model

Never trust accuracy alone.

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Look for:

* Precision
* Recall
* F1-score

Ask yourself:

* Are false negatives costly?
* Is imbalance affecting performance?

---

# Step 6: Try Another Model (Model Selection)

Try:

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print("DT Accuracy:", accuracy_score(y_test, dt.predict(X_test)))
```

Compare models.

Industry mindset:

> Never assume one model is best.

---

# Step 7: Save Model (Important)

You cannot retrain every time.

```python
import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
```

---

# Step 8: Deployment (Minimal Version)

Create simple prediction function.

```python
def predict_placement(cgpa, iq):
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    
    input_data = scaler.transform([[cgpa, iq]])
    prediction = model.predict(input_data)[0]
    
    return "Placed" if prediction == 1 else "Not Placed"
```

---

# Final Usage

```python
cgpa_input = float(input("Enter CGPA: "))
iq_input = float(input("Enter IQ: "))

result = predict_placement(cgpa_input, iq_input)

print("Prediction:", result)
```

Now you have:

Input → Model → Output

Just like CRUD flow.

---

# Now Let Me Be Honest

This is NOT impressive for placements.

This teaches:

* Data pipeline
* Scaling
* Model training
* Evaluation
* Saving model
* Inference pipeline

That’s the real goal.

---

# What This Toy Project Teaches You

1. End-to-end ML lifecycle
2. Separation of train vs inference
3. Importance of scaling
4. Evaluation mindset
5. Saving artifacts
6. Clean pipeline thinking

---

# What This Project Lacks (Industry Reality)

* No real dataset
* No missing values handling
* No imbalance handling
* No hyperparameter tuning
* No cross validation
* No monitoring
* No feature engineering

This is your ML “Hello World”.

---

# Next Level Upgrade (When Ready)

Add:

* Polynomial features
* ROC curve
* Cross validation
* GridSearchCV
* FastAPI deployment
* Dockerize

But not now.

Master pipeline first.

---

Now I’ll push you:

If CGPA and IQ are strongly correlated, what problem can occur in Logistic Regression?

Think.
Answer that before we move ahead.
