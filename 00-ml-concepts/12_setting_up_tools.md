# ML Environment & Tools Setup Guide

(Industry-focused, practical, no nonsense)

---

# 1️⃣ Anaconda & Jupyter Notebook

## What is Anaconda?

![Image](https://foundations-of-scientific-computing.readthedocs.io/en/latest/_images/navigator_2025.png)

![Image](https://i.sstatic.net/KLHi2.png)

![Image](https://nbclassic.readthedocs.io/en/latest/_images/jupyter-notebook-default.png)

![Image](https://docs.jupyter.org/en/stable/_images/trynb.png)

**Anaconda** is a Python distribution built for:

* Data science
* Machine learning
* Scientific computing

It comes with:

* Python
* Conda (environment manager)
* Preinstalled scientific libraries
* Jupyter Notebook

### Why use Anaconda?

Because:

* Dependency hell is real.
* Different projects need different library versions.
* Conda environments isolate projects.

If you don’t isolate environments, you will break things.

---

## Core Concept: Virtual Environments

Every serious ML engineer uses environments.

Example:

```bash
conda create -n ml_env python=3.10
conda activate ml_env
```

Then install only what that project needs:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Why this matters:

* Reproducibility
* Collaboration
* Deployment compatibility

In industry, this is non-negotiable.

---

## Jupyter Notebook

Used for:

* Exploration
* Visualization
* Prototyping
* Teaching

Not ideal for:

* Large production systems
* Complex pipelines
* Deployment code

Use it for:

* EDA
* Experiments
* Trying models

Move serious code to `.py` modules later.

---

# 2️⃣ Jupyter Notebook Tricks (That Save Time)

## 1. Keyboard Shortcuts (Mandatory)

* `A` → Add cell above
* `B` → Add cell below
* `Shift + Enter` → Run cell
* `DD` → Delete cell
* `M` → Markdown
* `Y` → Code

If you're clicking buttons with mouse all the time, you're wasting time.

---

## 2. Magic Commands

```python
%timeit some_function()
%matplotlib inline
```

Used for:

* Performance measurement
* Inline visualization

---

## 3. Restart Kernel Often

Memory leaks happen.
State pollution happens.

If notebook runs only when executed in order — you're writing bad notebook logic.

---

# 3️⃣ Using Kaggle

![Image](https://i.imgur.com/43ho0EG.png)

![Image](https://serokell.io/files/wb/wbiw1wd4.Frame_4_%281%29.png)

![Image](https://image.ibb.co/j9Ybwz/Screenshot_from_2018_10_05_19_47_35.png)

![Image](https://i.imgur.com/mOG3u3Z.png)

## What is Kaggle?

Kaggle is a platform for:

* ML competitions
* Public datasets
* Notebooks
* Community code

---

## Why Kaggle Is Important

* Real datasets
* Real ML problems
* Leaderboard pressure
* Production-like constraints

If you never touch Kaggle, you’re not testing yourself.

---

## How to Use Kaggle Properly

1. Download dataset
2. Do EDA
3. Baseline model
4. Improve step-by-step
5. Document learnings

Do NOT:

* Copy top solution
* Jump to complex models
* Ignore data understanding

---

# 4️⃣ Google Colab (GPU Usage for Deep Learning)

![Image](https://www.researchgate.net/publication/367403291/figure/fig2/AS%3A11431281390458142%401745273612054/Screen-capture-of-the-Google-Colab-interface.tif)

![Image](https://i.sstatic.net/kpPEmcb8.jpg)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2Au49MRjfN1dPu-i50ltK0bw.png)

![Image](https://www.tutorialspoint.com/google_colab/images/enabling_gpu.jpg)

## What is Google Colab?

Google Colab is a free cloud Jupyter environment.

Key advantage:

* Free GPU
* No setup
* Shareable notebooks

---

## How to Enable GPU

Runtime → Change runtime type → Hardware accelerator → GPU

Then verify:

```python
import torch
torch.cuda.is_available()
```

If False → you're on CPU.

---

## When to Use Colab

Use Colab when:

* Training deep learning models
* Handling large image datasets
* Your local machine is weak

Do NOT depend on Colab forever.
Industry uses:

* AWS
* GCP
* Azure

---

# 5️⃣ Running Large Kaggle Datasets on Colab (Images / Deep Learning)

### Step 1: Get Kaggle API

Install:

```bash
pip install kaggle
```

Upload Kaggle API key (`kaggle.json`) to Colab.

Then:

```bash
kaggle datasets download -d dataset-name
```

Unzip:

```bash
unzip file.zip
```

---

## Important: Memory Awareness

Colab free tier:

* Limited RAM
* Limited GPU hours

If you load entire image dataset in memory → crash.

Use:

* Data generators
* Batch loading
* PyTorch Dataset class
* TensorFlow Data pipelines

This is where beginners fail.

---

# 6️⃣ Famous ML & Data Science Tools (Industry View)

## Programming

* Python
* SQL

## Core Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* PyTorch
* TensorFlow

## Data Versioning

* DVC

## Experiment Tracking

* MLflow
* Weights & Biases

## Deployment

* FastAPI
* Docker
* Streamlit

## Cloud

* AWS
* GCP
* Azure

If your goal is 12–15 LPA in Bangalore, you must know:

* ML fundamentals
* Model deployment basics
* Some cloud exposure

Just building notebooks is not enough.

---

# 7️⃣ Platforms to Practice ML (Like LeetCode for ML)

There is no perfect “LeetCode for ML”. ML is not algorithm puzzle solving.

But these are good:

---

## 1. Kaggle (Competitions)

Best for:

* Applied ML
* Feature engineering
* Real datasets

---

## 2. DrivenData

DrivenData

Focus:

* Social impact ML
* Real-world problems

---

## 3. Zindi

Zindi

Emerging markets datasets.
Good practice for messy data.

---

## 4. Papers With Code

Papers With Code

For:

* SOTA models
* Benchmark comparisons
* Research understanding

Not for beginners. Use later.

---

## 5. UCI Machine Learning Repository

UCI Machine Learning Repository

Classic datasets for:

* Understanding fundamentals
* Clean academic examples

---

# 8️⃣ What I Recommend For YOU (Based on Your ML Goal)

You are building `applied-machine-learning` repo.

So your workflow should be:

### Phase 1: Foundations

* Anaconda
* Jupyter
* Local datasets
* Small CSV datasets

### Phase 2: Real Datasets

* Kaggle datasets
* Structured problems

### Phase 3: Deep Learning

* Colab GPU
* Image datasets
* Batch loading
* Experiment tracking

### Phase 4: Production Thinking

* Modular Python scripts
* Version control
* Model serialization
* Basic deployment

---

# Brutal Truth

Most people:

* Install Anaconda
* Run 3 notebooks
* Watch tutorials
* Say “I know ML”

But they cannot:

* Handle large datasets
* Debug memory errors
* Structure experiments
* Deploy models
* Write clean code

We won’t do that.

---

Now I want you to answer honestly:

1. What machine do you have? RAM? GPU?
2. Have you used virtual environments properly before?
3. Have you ever trained a deep learning model with GPU?
4. Do you want to set up local-first or cloud-first workflow?

Answer these clearly.
We will design your setup strategically.
