# Working with CSV Files in Pandas

---

## 1. Intuition

A CSV file is just a **text file** where values are separated by a delimiter (usually a comma).

Example:

```
id,name,age,salary
1,Akshit,22,40000
2,Rahul,25,50000
```

Pandas converts this text file into a **DataFrame** (tabular structure).

If you load data incorrectly:

* columns shift
* datatypes break
* missing values get misinterpreted
* memory explodes for large files

So loading data properly is **data engineering skill**, not just syntax.

---

## 2. Importing Pandas

```python
import pandas as pd
```

Industry standard alias is `pd`.

---

## 3. Opening a Local CSV File

```python
df = pd.read_csv("data.csv")
```

This loads file from current working directory.

If file is in another folder:

```python
df = pd.read_csv("datasets/sales.csv")
```

---

## 4. Opening CSV from URL

```python
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)
```

Used when:

* downloading public datasets
* APIs returning CSV
* remote storage (S3 links etc.)

---

## 5. `sep` Parameter (Separator)

### Intuition

Default separator is `,`.

But sometimes data uses:

* `;`
* `|`
* `\t` (tab)

Example:

```
id;name;age
1;Akshit;22
```

```python
df = pd.read_csv("data.csv", sep=";")
```

If you don't set correct separator → entire row becomes single column.

---

## 6. `index_col` Parameter

### Intuition

By default, pandas creates index: 0,1,2,3...

If your dataset already has a unique ID column:

```python
df = pd.read_csv("data.csv", index_col="id")
```

Now `id` becomes index.

Why useful?

* Faster lookup
* Cleaner table
* Avoid duplicate ID column

---

## 7. `header` Parameter

### Intuition

Header tells pandas which row contains column names.

Default: first row (0)

If no header in file:

```
1,Akshit,22
2,Rahul,25
```

```python
df = pd.read_csv("data.csv", header=None)
```

Pandas assigns numeric column names: 0,1,2...

If header is in 2nd row:

```python
df = pd.read_csv("data.csv", header=1)
```

---

## 8. `usecols` Parameter

### Intuition

Sometimes dataset has 200 columns but you need only 5.

Load only required columns to:

* reduce memory
* speed up processing

```python
df = pd.read_csv("data.csv", usecols=["name", "salary"])
```

Or by index:

```python
df = pd.read_csv("data.csv", usecols=[0,2,3])
```

Industry rule: **Never load unnecessary columns in production.**

---

## 9. `squeeze` Parameter (Deprecated in new versions)

Earlier:

```python
series = pd.read_csv("data.csv", squeeze=True)
```

If file had one column → returned Series instead of DataFrame.

Now recommended:

```python
df = pd.read_csv("data.csv")
series = df.iloc[:, 0]
```

---

## 10. `skiprows` and `nrows`

### skiprows

Skip unwanted rows:

```python
df = pd.read_csv("data.csv", skiprows=2)
```

Or specific rows:

```python
df = pd.read_csv("data.csv", skiprows=[1,3])
```

---

### nrows

Load only first N rows:

```python
df = pd.read_csv("data.csv", nrows=100)
```

Useful for:

* testing
* huge datasets

---

## 11. `encoding` Parameter

### Intuition

CSV is text. Text encoding matters.

Common encodings:

* `utf-8` (default)
* `latin1`
* `ISO-8859-1`

If you see error:

```
UnicodeDecodeError
```

Try:

```python
df = pd.read_csv("data.csv", encoding="latin1")
```

Very common in real-world datasets.

---

## 12. Skip Bad Lines

If dataset contains corrupted rows:

```python
df = pd.read_csv("data.csv", on_bad_lines="skip")
```

Options:

* "error"
* "warn"
* "skip"

Production note:
Skipping silently may hide data issues. Always log.

---

## 13. `dtype` Parameter

### Intuition

By default pandas guesses datatype.

Sometimes:

* numeric column becomes object
* ID becomes int but should be string

Example:

```python
df = pd.read_csv("data.csv", dtype={"id": str})
```

Why important?

If ID = 00123
Without string → becomes 123 (loses leading zeros)

In ML pipelines → this can break joins.

---

## 14. Handling Dates

CSV stores dates as strings.

Convert while loading:

```python
df = pd.read_csv("data.csv", parse_dates=["order_date"])
```

Or multiple columns:

```python
df = pd.read_csv("data.csv", parse_dates=["start_date", "end_date"])
```

After loading:

```python
df["order_date"].dt.year
```

Production importance:
Time-based features depend on proper datetime conversion.

---

## 15. `converters` Parameter

### Intuition

Apply function while loading data.

Example: salary with ₹ symbol

```
₹40,000
```

```python
df = pd.read_csv(
    "data.csv",
    converters={"salary": lambda x: int(x.replace("₹","").replace(",",""))}
)
```

Useful when:

* cleaning required during load
* data not clean

But heavy converters slow down loading.

---

## 16. `na_values` Parameter

### Intuition

Different datasets represent missing values differently:

* NA
* N/A
* ?
* *
* null

```python
df = pd.read_csv("data.csv", na_values=["NA", "?", "-"])
```

Now pandas converts them to `NaN`.

Critical for:

* correct missing value handling
* avoiding wrong model training

---

## 17. Loading Huge Dataset in Chunks

### Problem

If file = 5GB
Your RAM = 8GB
System crashes.

### Solution → Chunking

```python
chunk_size = 100000

for chunk in pd.read_csv("huge.csv", chunksize=chunk_size):
    print(chunk.shape)
```

Each chunk is a DataFrame.

Use case:

* Aggregation
* Filtering
* Preprocessing large data

Example:

```python
total = 0

for chunk in pd.read_csv("huge.csv", chunksize=100000):
    total += chunk["sales"].sum()

print(total)
```

Industry tip:
For very large data → use:

* Dask
* Spark
* Polars

Not plain pandas.

---

# Practical Demo Dataset Example

Let’s assume dataset:

```
sales_data.csv
```

Columns:

* order_id
* customer_id
* order_date
* product
* quantity
* price
* region

### Production-ready loading example:

```python
df = pd.read_csv(
    "sales_data.csv",
    parse_dates=["order_date"],
    dtype={"order_id": str, "customer_id": str},
    na_values=["NA", "null", "?"],
    usecols=["order_id","customer_id","order_date","quantity","price"],
    encoding="utf-8"
)
```

This is how an ML engineer loads data safely.

---

# Common Beginner Mistakes

1. Ignoring encoding errors
2. Not checking dtypes
3. Loading full dataset without checking size
4. Not using parse_dates
5. Using converters blindly
6. Not handling missing values
7. Hardcoding absolute paths

---

# Production-Level Thinking

Before loading dataset, always ask:

* File size?
* Memory available?
* Required columns?
* Correct data types?
* Date parsing needed?
* Missing values format?
* Is data clean or corrupted?
* Should we validate schema?

If you don’t think about these, you are acting like a notebook coder, not an ML engineer.

---