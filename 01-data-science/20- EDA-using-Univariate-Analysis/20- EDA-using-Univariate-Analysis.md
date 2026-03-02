# 📊 EDA – Univariate Analysis

---

## 1️⃣ What is EDA?

**Exploratory Data Analysis (EDA)** is the process of:

* Understanding structure of data
* Detecting patterns
* Finding anomalies
* Checking assumptions
* Identifying relationships

Before training any ML model.

If you skip this, you are just doing blind modeling.

---

## 2️⃣ Types of EDA

### 1. Univariate Analysis

→ Analyze **one variable at a time**

Example:

* Distribution of Age
* Count of male vs female

### 2. Bivariate Analysis

→ Relationship between **two variables**

Example:

* Age vs Survival
* Gender vs Survival

### 3. Multivariate Analysis

→ Relationship among **multiple variables**

Example:

* Age + Class + Gender → Survival

---

# 🔹 Univariate Analysis

We analyze:

* One column
* Distribution
* Spread
* Shape
* Outliers
* Missing values

Two main data types:

1. Categorical
2. Numerical

---

# 🚢 Sample Dataset: Titanic

![Image](https://www.casetalk.com/images/stories/titanic/titanic-data.png)

![Image](https://i.imgur.com/AC9Bq63.png)

![Image](https://community.fabric.microsoft.com/oxcrx34285/attachments/oxcrx34285/DataStoriesGallery/8707/1/DataDNA%20Dataset%20Challenge%20October%202022%20-%20Sarita%20S-01.jpg)

![Image](https://miro.medium.com/1%2AqTqs-DD_i5dqUtTuaVzfWw.png)

Common Columns:

* Survived (0/1)
* Pclass (1,2,3)
* Sex (male/female)
* Age
* Fare
* Embarked

---

# 🟢 1. Univariate Analysis – Categorical Data

Categorical = categories or labels

Examples:

* Sex
* Embarked
* Pclass
* Survived

---

## What Do We Check?

* Frequency count
* Class imbalance
* Missing categories
* Rare categories

---

## A) Countplot

Most commonly used in ML.

### Why?

Because ML models care about class distribution.

### Example: Sex

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Sex', data=df)
plt.show()
```

### What It Shows:

* Number of males
* Number of females

### What To Observe:

* Is dataset imbalanced?
* Any rare category?
* Missing categories?

---

## B) Pie Chart

```python
df['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
```

### Honest Opinion

Pie charts look nice.

But in ML practice?
Rarely used.

Countplot is more informative and readable.

---

## C) Bar Plot (Value Counts)

```python
df['Embarked'].value_counts().plot(kind='bar')
plt.show()
```

Used when:

* Categories are many
* Need sorted frequencies

---

## D) Frequency Table (Very Important)

Before plotting:

```python
df['Pclass'].value_counts()
df['Pclass'].value_counts(normalize=True)
```

Always check numbers first.
Graphs come second.

Industry tip:
If you only show plots in interview without numbers,
it looks shallow.

---

## ⚠ What Problems We Detect in Categorical?

* Class imbalance
* Rare category (can hurt model)
* Missing values
* Wrong labels (e.g. typo)

---

# 🔵 2. Univariate Analysis – Numerical Data

Numerical = measurable values

Examples:

* Age
* Fare
* SibSp
* Parch

---

## What Do We Check?

* Distribution shape
* Skewness
* Spread
* Outliers
* Central tendency
* Missing values

---

## A) Histogram

Most fundamental visualization.

```python
plt.hist(df['Age'], bins=20)
plt.show()
```

### What It Shows:

* Distribution shape
* Skewed or symmetric?
* Multiple peaks?

---

## B) Distplot (Old) / KDE Plot (Better)

```python
sns.kdeplot(df['Age'], fill=True)
plt.show()
```

Shows smooth density curve.

Better for understanding shape.

---

## C) Boxplot (VERY IMPORTANT)

```python
sns.boxplot(x=df['Fare'])
plt.show()
```

### Why Important?

Detects:

* Outliers
* Median
* Spread (IQR)

Industry loves boxplots.

---

## D) Summary Statistics

Before plotting:

```python
df['Age'].describe()
```

Gives:

* Mean
* Std
* Min
* 25%
* 50%
* 75%
* Max

Plot + numbers together = strong EDA

---

# 📈 Additional Useful Plots for Numerical

### 1. Violin Plot

```python
sns.violinplot(x=df['Age'])
```

Shows distribution + density + quartiles.

---

### 2. Rug Plot

Shows individual data points.

Mostly academic.

---

# 🎯 What You Must Observe in Numerical Data

1. Is it normally distributed?
2. Is it heavily skewed?
3. Are there extreme outliers?
4. Is transformation needed? (log transform)
5. Are there impossible values? (negative age?)

---

# 🚨 Common Beginner Mistakes

1. Only plotting, not interpreting
2. Ignoring missing values
3. Ignoring skewness
4. Not checking outliers
5. Blindly removing outliers
6. Using pie charts for everything

---

# 🏭 Production Thinking

Ask:

* Will this distribution change over time?
* Are outliers real or data entry error?
* Will class imbalance affect model training?
* Should I apply scaling?
* Should I apply transformation?

Example:
Fare is highly right-skewed in Titanic.
If you apply linear regression without log transform,
model may behave badly.

---

# 🧠 Real ML Insight

Univariate analysis helps you decide:

* Feature scaling needed?
* Encoding method?
* Log transform?
* Drop feature?
* Imputation strategy?

If you skip this,
your preprocessing will be random.

---

# 🔥 Summary

Univariate Analysis =

For Categorical:

* Countplot
* Bar plot
* Frequency table

For Numerical:

* Histogram
* KDE
* Boxplot
* Summary stats

Always:

Numbers first
Plots second
Interpretation third

---

Now I’ll challenge you.

If Age is right skewed:

* What transformation would you apply?
* Why?
* When would you NOT apply it?

Answer logically. No guessing.
