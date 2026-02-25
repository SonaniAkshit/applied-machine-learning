# Tensors

---

# 1. Intuition (What Is a Tensor Really?)

Forget the heavy math for a moment.

A **tensor is just a container for numbers arranged in a structured way.**

That’s it.

The only difference between:

* a single number
* a list of numbers
* a table of numbers
* a stack of images
* a batch of videos

is how many dimensions they have.

That structure is what we call a **tensor**.

---

# 2. From Scalar → Vector → Matrix → n-D Tensor

Let’s build step by step.

---

## 🔹 0D Tensor (Scalar)

A single number.

Example:

```
5
```

That’s a tensor with:

* ndim = 0
* shape = ()

Think:

* Temperature today = 32
* Student CGPA = 8.1

That’s a scalar.

---

## 🔹 1D Tensor (Vector)

A list of numbers.

Example:

```
[8.1, 91, 1, 1]
```

This could represent one student:

| CGPA | IQ | STATE | PLACED |
| ---- | -- | ----- | ------ |
| 8.1  | 91 | 1     | 1      |

Here:

* ndim = 1
* shape = (4,)
* 4 features

This is a **feature vector**.

---

## 🔹 2D Tensor (Matrix)

A table of numbers.

Example dataset of 3 students:

```
[
 [8.1, 91, 1, 1],
 [7.2, 85, 0, 0],
 [9.0, 95, 1, 1]
]
```

Now:

* ndim = 2
* shape = (3, 4)

  * 3 rows (students)
  * 4 columns (features)

This is how tabular ML datasets are stored.

---

### Visual Representation of 2D Tensor

![Image](https://www.slideteam.net/media/catalog/product/cache/1280x720/3/x/3x4_matrix_table_showing_text_boxes_Slide01.jpg)

![Image](https://math.hws.edu/javanotes/c7/two-dimensional-array.png)

![Image](https://i.pinimg.com/736x/37/9d/c6/379dc67f47e8dd0986608297ed9901a0.jpg)

![Image](https://www.researchgate.net/publication/371858352/figure/fig5/AS%3A11431281170446956%401687781598360/sualization-of-a-matrix-with-numbers-This-way-of-representing-a-matrix-is-usually.png)

Rows = samples
Columns = features

Industry rule:

* Rows → observations
* Columns → features

---

## 🔹 3D Tensor

Now imagine:

You don’t have just student data.

You have grayscale images.

A grayscale image is:

* height
* width
* pixel intensity

Example:

* 28 × 28 image

Shape:

```
(28, 28)
```

Now suppose you have 100 images.

Shape becomes:

```
(100, 28, 28)
```

Now it is a **3D tensor**.

---

### Visual Example of 3D Tensor (Stack of Images)

![Image](https://api.wandb.ai/files/vincenttu/images/projects/37163867/6e8f935a.png)

![Image](https://www.researchgate.net/publication/262885990/figure/fig9/AS%3A668297364242447%401536345916765/a-exemplifies-the-process-of-unfolding-a-3D-tensor-There-are-three-matrix.ppm)

![Image](https://miro.medium.com/1%2A0pRV1-1uNUxLmlvooreVpw.png)

![Image](https://content-media-cdn.codefinity.com/courses/a668a7b9-f71f-420f-89f1-71ea7e5abbac/Ch.%2B1-2/nd_tensor.png)

Think of it as:
Stack of matrices.

---

## 🔹 4D Tensor (Very Common in Deep Learning)

Now real-world CNN case.

For color images:

Each image has:

* height
* width
* channels (RGB)

So one image:

```
(224, 224, 3)
```

Now batch of 32 images:

```
(32, 224, 224, 3)
```

This is a 4D tensor.

Meaning:

* 32 images
* each 224 × 224
* 3 color channels

This is exactly what CNNs expect.

---

### Visual of 4D Tensor (Batch of RGB Images)

![Image](https://www.researchgate.net/publication/380597911/figure/fig4/AS%3A11431281263218649%401721959791699/RGB-color-space-3D-coordinate-diagram-and-visualization-of-each-channel.png)

![Image](https://discuss.pytorch.org/uploads/default/original/3X/0/e/0e1081af545aa5fa768ac7af0765881a0c9cece9.png)

![Image](https://deeplizard.com/assets/jpg/7b13a505.jpg)

![Image](https://i.sstatic.net/TKea8.png)

---

## 🔹 5D Tensor (Videos)

Video = sequence of images.

So now dimensions become:

```
(batch, frames, height, width, channels)
```

Example:

```
(16, 30, 224, 224, 3)
```

Meaning:

* 16 videos
* 30 frames each
* 224 × 224
* RGB

That’s a 5D tensor.

---

# 3. Important Concepts

---

## Rank (or ndim)

Rank = number of axes.

| Tensor Type | Rank |
| ----------- | ---- |
| Scalar      | 0    |
| Vector      | 1    |
| Matrix      | 2    |
| Image stack | 3    |
| Batch image | 4    |
| Video batch | 5    |

Simple.

Rank = number of brackets you need to access a number.

---

## Shape

Shape tells you:

How many elements exist in each dimension.

Example:

```
(3, 4)
```

Means:

* 3 rows
* 4 columns

Example:

```
(32, 224, 224, 3)
```

Means:

* 32 images
* 224 height
* 224 width
* 3 channels

---

## Axis

Axis = direction of movement.

In (3,4):

* axis 0 → rows
* axis 1 → columns

In (32,224,224,3):

* axis 0 → batch
* axis 1 → height
* axis 2 → width
* axis 3 → channel

Understanding axis is critical for:

* summation
* normalization
* batch operations

---

# 4. Real ML Dataset Examples

---

## Example 1: Student Placement Dataset (Tabular)

Suppose:

10,000 students
5 features each

Shape:

```
(10000, 5)
```

Used in:

* Logistic Regression
* Decision Trees
* XGBoost

---

## Example 2: MNIST Dataset

MNIST:

* 60,000 grayscale images
* 28 × 28

Tensor shape:

```
(60000, 28, 28)
```

If batch size = 64:

```
(64, 28, 28)
```

---

## Example 3: CIFAR-10

* 60,000 color images
* 32 × 32
* RGB

Shape:

```
(60000, 32, 32, 3)
```

CNN input:

```
(batch, 32, 32, 3)
```

---

# 5. Why Tensors Matter in ML

Every ML model expects input in tensor format.

Neural network does:

Input tensor → matrix multiplication → activation → output tensor

If you don’t understand shape, you will constantly get:

* shape mismatch error
* broadcasting error
* wrong training

This is not optional knowledge.

---

# 6. Beginner Mistakes

1. Confusing shape and size
2. Thinking 3D means “3D object”
3. Ignoring batch dimension
4. Not understanding axis in operations
5. Hardcoding shapes

---

# 7. Production-Level Thinking

Now important.

In production:

You must know:

* How much memory this tensor takes
* How to reshape safely
* When to use float32 vs float64
* How batching affects GPU usage

Example:

Tensor:

```
(100000, 224, 224, 3)
```

This can crash GPU memory.

So we use batching:

```
(32, 224, 224, 3)
```

Engineers think about:

* scalability
* memory footprint
* parallelization

Not just theory.

---

# 8. Mathematical View (Simple)

A tensor is:

A multi-dimensional array.

You can think of it as:

[
T \in \mathbb{R}^{d_1 \times d_2 \times \dots \times d_n}
]

Where:

* ( d_1, d_2, \dots, d_n ) are dimensions
* n = rank

Example:
[
T \in \mathbb{R}^{3 \times 4}
]

Means 3 rows, 4 columns.

---

# Final Mental Model

Scalar → single number
Vector → list
Matrix → table
3D → stack of tables
4D → batch of stacks
5D → batch of videos

Nothing magical.

Just structured numbers.
