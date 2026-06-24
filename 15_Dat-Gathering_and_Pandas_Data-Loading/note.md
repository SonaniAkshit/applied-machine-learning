# Data Gathering and Pandas Data Loading

---

## 1. Data Gathering Overview

### Data Gathering Methods

```mermaid
graph TD
    DG["🔧 Data Gathering"]

    DG --> CSV["📄 CSV Files"]
    DG --> JSON["🗄️ JSON/SQL"]
    DG --> API["🌐 APIs fetch"]
    DG --> WS["🕷️ Web Scraping"]

    LF["🖥️ Local Files"] --> CSV
    URL["🔗 URLs"] --> CSV

    SD["📊 Structured Data"] --> JSON
    Q["🔍 Queries"] --> JSON

    REST["🔌 REST/GraphQL"] --> API
    EP["🎯 Endpoints"] --> API

    HTML["</> HTML Parsing"] --> WS
    DC["⚡ Dynamic Content"] --> WS

    OP["🖥️ Online Platforms"] --> SF["📋 Surveys & Forms"]
    MI["⌨️ Manual Input"] --> SF
    SF --> DG
```

> **Diagram Explanation:** This diagram shows the four primary data gathering methods — CSV Files, JSON/SQL, APIs, and Web Scraping — each with their respective input sources. CSV files can be sourced from local files or URLs; JSON/SQL from structured data or queries; APIs via REST/GraphQL endpoints; and Web Scraping via HTML parsing or dynamic content.

---

### Data Source Comparison

| Method | Use Case | Complexity | Real-time | Example |
|---|---|---|---|---|
| CSV | Structured tabular data | Low | No | Training datasets |
| JSON/SQL | Nested/relational data | Medium | Depends | Database exports |
| API | Dynamic data | Medium-High | Yes | Twitter, Weather |
| Web Scraping | Unstructured web data | High | Depends | Product prices |

---

## 2. Sample Datasets Overview

### Dataset 1: Training Data (aug_train.csv)

**Characteristics:**

- Employee enrollment data
- Multiple features: city, gender, experience, education
- Target variable for prediction

#### Sample Data Structure

| Column | Type | Example | Purpose |
|---|---|---|---|
| enrollee_id | Integer | 8949 | Unique identifier |
| city | Categorical | city_103 | Location |
| city_development_index | Float | 0.92 | City development score |
| gender | Categorical | Male/Female | Demographics |
| relevent_experience | Categorical | Has/No experience | Work history |
| enrolled_university | Categorical | Full time/Part time/no_enrollment | Education status |
| education_level | Categorical | Graduate/Masters/High School | Education |
| major_discipline | Categorical | STEM/Business/Arts | Field of study |
| experience | Categorical | >20, 15, 5, never | Years of experience |
| company_size | Categorical | 50-99, 1000-4999 | Company size |
| company_type | Categorical | Pvt Ltd, Funded Startup | Company type |
| last_new_job | Categorical | 1, 2, >4, never | Job change recency |
| training_hours | Integer | 36, 47, 83 | Hours of training |
| target | Binary | 0, 1 | Target variable |

---

### Dataset 2: Movie Metadata (movie_titles_metadata.tsv)

**Characteristics:**

- Tab-separated values (TSV)
- Movie information with genres, ratings
- Multiple features per movie

#### Sample Movie Data

| Field | Example | Description |
|---|---|---|
| ID | m0, m1, m2 | Unique movie identifier |
| Title | "10 things i hate about you" | Movie name |
| Year | 1999 | Release year |
| Rating | 6.90 | User rating (0-10) |
| Votes | 62847 | Number of votes |
| Genres | ['comedy', 'romance'] | Genre list |

---

## 3. Pandas Data Loading Methods

### Method Hierarchy

```mermaid
graph TD
    PD["pandas Data Loading"]
    PD --> LC["Local CSV\npd.read_csv('file.csv')"]
    PD --> URL["From URL\nrequests + StringIO"]
    PD --> TSV["TSV File\nsep='\\t'"]
    PD --> OPT["Advanced Parameters"]

    OPT --> IDX["index_col"]
    OPT --> HDR["header"]
    OPT --> SKP["skiprows / nrows"]
    OPT --> ENC["encoding"]
    OPT --> DTP["dtype"]
    OPT --> CNV["converters"]
    OPT --> NAV["na_values"]
    OPT --> DT["parse_dates"]
    OPT --> CHK["chunksize"]
```

> **Diagram Explanation:** This hierarchy shows that Pandas data loading branches into loading from local CSV, from URLs, from TSV files, and via advanced parameters. The advanced parameters sub-branch enumerates the key `read_csv()` options covered throughout these notes.

---

## 4. Comprehensive `pandas.read_csv()` Parameters

### 1. Import Pandas

```python
import pandas as pd
```

**Purpose:** Essential library for data manipulation

---

### 2. Opening a Local CSV File

```python
df = pd.read_csv('aug_train.csv')
```

**Basic Usage:**

- Reads CSV from local file system
- Auto-detects separator (comma by default)
- Returns DataFrame object

---

### 3. Opening CSV from URL

```python
import requests
from io import StringIO

url = ""
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.
req = requests.get(url, headers=headers)
data = StringIO(req.text)

pd.read_csv(data)
```

**Use Case:** Fetch data directly from web URLs

---

### 4. `sep` Parameter (Separator)

```python
pd.read_csv('movie_titles_metadata.tsv', sep='\t')
```

**Purpose:** Specify delimiter for non-comma separated files

| Separator | Code | File Type | Example |
|---|---|---|---|
| Comma | `sep=','` | CSV | Standard CSV |
| Tab | `sep='\t'` | TSV | Tab-separated |
| Semicolon | `sep=';'` | European CSV | Excel exports |
| Pipe | `` `sep='` `` | `\|` | Custom |
| Whitespace | `sep='\s+'` | Space-delimited | Fixed-width |

**Advanced Usage:**

```python
# Multiple columns with names
pd.read_csv('movie_titles_metadata.tsv',
            sep='\t',
            names=['sno', 'name', 'release_year', 'rating', 'votes', 'genres']
```

---

### 5. `index_col` Parameter

```python
pd.read_csv('aug_train.csv', index_col='enrollee_id')
```

**Purpose:** Set specific column as DataFrame index

```mermaid
graph LR
    A["Default Index\n0, 1, 2, 3..."] -- "index_col" --> B["Custom Index\nenrollee_id"]
```

> **Diagram Explanation:** This diagram illustrates how applying `index_col` transforms the default integer-based index (0, 1, 2, 3…) into a meaningful custom index using the `enrollee_id` column.

**Comparison:**

| Without `index_col` | With `index_col='enrollee_id'` |
|---|---|
| Index: 0, 1, 2... | Index: 8949, 29725, 11561... |
| enrollee_id is column | enrollee_id is index |
| Need df['enrollee_id'] | Access via df.loc[8949] |

**Benefits:**

- Faster lookups by ID
- Cleaner data representation
- Direct row access: `df.loc[enrollee_id]`

---

### 6. `header` Parameter

```python
pd.read_csv('test.csv', header=1)
```

**Purpose:** Specify which row contains column names

| Value | Meaning | Use Case |
|---|---|---|
| `header=0` | First row (default) | Standard CSV |
| `header=1` | Second row | Skip title row |
| `header=None` | No header | Generate numeric columns |
| `header=[0,1]` | Multi-level | Hierarchical columns |

**Example Scenarios:**

```python
# No header - pandas creates numeric columns
df = pd.read_csv('test.csv', header=None)
# Columns: 0, 1, 2, 3...

# Custom header
df = pd.read_csv('test.csv', names=['col1', 'col2', 'col3'], header=0)
```

---

### 7. `skiprows` / `nrows` Parameter

```python
pd.read_csv('aug_train.csv', skiprows=[0, 5])
```

**Purpose:** Skip specific rows during loading

**Usage Patterns:**

| Parameter | Example | Result |
|---|---|---|
| `skiprows=1` | Skip first row | Start from row 2 |
| `skiprows=[0,5]` | Skip rows 0 and 5 | Selective skipping |
| `skiprows=range(1,5)` | Skip rows 1-4 | Range skipping |
| `nrows=100` | Read first 100 | Limit data load |
| `skiprows=lambda x: x % 2 == 0` | Skip even rows | Conditional skipping |

**Sample Size for Testing:**

```python
# Quick data exploration
df_sample = pd.read_csv('large_file.csv', nrows=1000)
```

---

### 8. `encoding` Parameter

```python
pd.read_csv('zomato.csv', encoding='latin-1')
```

**Purpose:** Handle different character encodings

#### Common Encodings

| Encoding | Use Case | Region | Characters |
|---|---|---|---|
| utf-8 | Default, universal | Global | All languages |
| latin-1 | Western Europe | Europe | Accented chars |
| cp1252 | Windows | Windows | Windows apps |
| iso-8859-1 | Western Europe | Europe | Similar to latin-1 |
| utf-16 | Unicode | Global | Asian languages |

**Common Issues:**

```python
# Error: UnicodeDecodeError
df = pd.read_csv('file.csv')  # Fails with special characters

# Solution
df = pd.read_csv('file.csv', encoding='latin-1')
```

---

### 9. Skip Bad Lines

```python
pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', error_bad_lines=False)
```

**Purpose:** Handle malformed rows gracefully

**Note:** In pandas 1.3+, use:

```python
pd.read_csv('file.csv', on_bad_lines='skip')  # New syntax
```

**Options:**

| Parameter | Behavior | When to Use |
|---|---|---|
| `error_bad_lines=False` | Skip bad lines | Corrupted data |
| `warn_bad_lines=True` | Show warnings | Debug issues |
| `on_bad_lines='skip'` | Skip (new syntax) | pandas 1.3+ |
| `on_bad_lines='warn'` | Warn and skip | Monitor quality |

---

### 10. `dtype` Parameter

```python
pd.read_csv('aug_train.csv', dtype={'target': int})
```

**Purpose:** Specify data types for columns

#### Memory Optimization Example

```python
# Default - high memory
df = pd.read_csv('large_file.csv')
print(df.memory_usage())  # May show 1GB+

# Optimized
df = pd.read_csv('large_file.csv', dtype={
    'id': 'int32',        # Instead of int64
    'category': 'category',  # Instead of object
    'value': 'float32'    # Instead of float64
})
print(df.memory_usage())  # May show 400MB
```

**Info Check:**

```python
df.info()  # Shows data types and memory usage
```

---

### 11. `converters` Parameter

```python
def rename(name):
    if name == "Royal Challengers Bangalore":
        return "RCB"
    else:
        return name

pd.read_csv('IPL Matches 2008-2020.csv', converters={'team1': rename})
```

**Purpose:** Apply custom transformation functions during loading

#### Use Cases for Converters

| Scenario | Function | Example |
|---|---|---|
| **Abbreviations** | Team name shortening | "Royal Challengers" → "RCB" |
| **Data cleaning** | Remove special chars | "$1,000" → 1000 |
| **Format conversion** | Parse custom dates | "Jan-2020" → datetime |
| **Category mapping** | Map to codes | "High" → 1, "Low" → 0 |

#### Advanced Converter Examples

```python
# Multiple converters
converters = {
    'price': lambda x: float(x.replace('$', '').replace(',', '')),
    'date': lambda x: pd.to_datetime(x, format='%d-%m-%Y'),
    'category': lambda x: x.lower().strip()
}

df = pd.read_csv('data.csv', converters=converters)
```

---

### 12. `na_values` Parameter

```python
pd.read_csv('file.csv', na_values=['?', '-', 'N/A', 'null', ''])
```

**Purpose:** Specify which values should be treated as missing (NaN)

#### Default NA Values in Pandas

pandas automatically recognizes these as NA:

- `NaN`, `nan`, `NA`
- `null`, `NULL`
- Empty strings
- `None`

#### Custom NA Values Table

| Data Source | Custom NA | Example |
|---|---|---|
| Surveys | `'-'`, `'N/A'`, `'Unknown'` | Age: `'-'` |
| Databases | `'null'`, `'\N'` | Name: null |
| Excel | `'#N/A'`, `'#VALUE!'` | Formula errors |
| Custom | `'?'`, `'999'`, `'Missing'` | Sentinel values |

**Advanced Usage:**

```python
# Different NA values per column
na_dict = {
    'age': ['?', '-1', '999'],
    'income': ['N/A', 'unknown'],
    'category': ['', 'none', 'other']
}

df = pd.read_csv('survey.csv', na_values=na_dict)
```

---

### 13. Handling Dates

```python
pd.read_csv('file.csv', parse_dates=['date_column'])
```

**Purpose:** Automatically parse date columns

#### Date Parsing Methods

| Method | Code | Result |
|---|---|---|
| Auto parse | `parse_dates=['date']` | Convert to datetime |
| Multiple columns | `parse_dates=['date1', 'date2']` | Multiple conversions |
| Combine columns | `parse_dates=[['year', 'month', 'day']]` | Create single date |
| Custom format | `date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y')` | Specific format |

#### Date Format Examples

```python
# Single column
df = pd.read_csv('sales.csv', parse_dates=['order_date'])

# Multiple columns
df = pd.read_csv('events.csv', parse_dates=['start_date', 'end_date'])

# Combine columns into one date
df = pd.read_csv('records.csv',
                 parse_dates=[['year', 'month', 'day']])
# Creates 'year_month_day' column

# Custom format
from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv('logs.csv', parse_dates=['timestamp'], date_parser=dateparse)
```

---

### 14. `chunksize` Parameter - Loading Large Datasets

```python
# Instead of loading entire file
chunk_iter = pd.read_csv('huge_file.csv', chunksize=10000)

for chunk in chunk_iter:
    # Process each chunk
    process(chunk)
```

**Purpose:** Load data in manageable pieces for memory efficiency

#### When to Use Chunksize

| Scenario | Chunksize | Strategy |
|---|---|---|
| Memory limited | 10,000–50,000 | Process in batches |
| Aggregations | 100,000 | Compute per chunk, combine |
| Filtering | 50,000 | Filter each chunk, concat |
| Database loading | 10,000 | Insert chunks to DB |

#### Practical Examples

**Example 1: Filtering Large Data**

```python
chunks = []
for chunk in pd.read_csv('huge.csv', chunksize=50000):
    filtered = chunk[chunk['age'] > 18]
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
```

**Example 2: Aggregation**

```python
total_sales = 0
for chunk in pd.read_csv('sales.csv', chunksize=10000):
    total_sales += chunk['amount'].sum()

print(f"Total Sales: {total_sales}")
```

**Example 3: Database Loading**

```python
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@localhost/db')

for chunk in pd.read_csv('data.csv', chunksize=5000):
    chunk.to_sql('table_name', engine, if_exists='append', index=False)
```

---

## 5. Complete Parameter Reference Table

### All `read_csv()` Parameters Summary

| Parameter | Type | Purpose | Example |
|---|---|---|---|
| **filepath** | str | File path or URL | `'data.csv'` |
| **sep** | str | Delimiter | `','`, `'\t'`, `';'` |
| **header** | int/list | Row(s) for column names | `0`, `None`, `[0,1]` |
| **names** | list | Column names | `['col1', 'col2']` |
| **index_col** | int/str | Column to use as index | `'id'`, `0` |
| **usecols** | list | Columns to load | `['col1', 'col3']` |
| **dtype** | dict | Data types | `{'age': int}` |
| **converters** | dict | Custom converters | `{'col': func}` |
| **skiprows** | int/list | Rows to skip | `[0, 5]`, `1` |
| **nrows** | int | Number of rows to read | `1000` |
| **na_values** | list/dict | NA value representations | `['?', 'N/A']` |
| **encoding** | str | File encoding | `'utf-8'`, `'latin-1'` |
| **parse_dates** | list | Columns to parse as dates | `['date_col']` |
| **chunksize** | int | Read in chunks | `10000` |
| **compression** | str | Compression type | `'gzip'`, `'zip'` |
| **error_bad_lines** | bool | Skip bad lines | `False` |
| **warn_bad_lines** | bool | Warn on bad lines | `True` |
| **low_memory** | bool | Memory usage | `True` |
| **memory_map** | bool | Memory mapping | `True` |

---

## 6. Best Practices and Optimization

### Loading Strategy Decision Tree

```mermaid
graph TD
    FS{"File Size?"}
    FS -- "< 100MB" --> SM["< 100MB"]
    FS -- "100MB - 1GB" --> MD["100MB - 1GB"]
    FS -- "> 1GB" --> LG["> 1GB"]

    SM --> LD["Load directly"]
    MD --> OD["Optimize dtypes"]
    LG --> UC["Use chunksize"]

    OD --> SD["Specify dtypes"]
    OD --> SU["Use usecols"]

    UC --> PB["Process in batches"]
    UC --> DV["Use Dask/Vaex"]
```

> **Diagram Explanation:** This decision tree guides file loading strategy based on file size. Small files (<100MB) can be loaded directly. Medium files (100MB–1GB) benefit from dtype optimization and column selection. Large files (>1GB) require chunked loading or out-of-core tools like Dask/Vaex.

---

### Optimization Checklist

**For Small Files (<100MB):**

- ✅ Use default `pd.read_csv()`
- ✅ Specify encoding if errors occur
- ✅ Set appropriate `index_col`

**For Medium Files (100MB-1GB):**

- ✅ Specify dtypes explicitly
- ✅ Use `usecols` to load only needed columns
- ✅ Convert object columns to category when possible
- ✅ Check memory with `df.info(memory_usage='deep')`

**For Large Files (>1GB):**

- ✅ Use `chunksize` parameter
- ✅ Process data in batches
- ✅ Consider Dask or Vaex for out-of-core processing
- ✅ Use database storage instead of loading entirely
- ✅ Filter early, keep only necessary rows

#### Memory Optimization Example

```python
# Before optimization
df = pd.read_csv('large.csv')
print(df.memory_usage(deep=True).sum() / 1024**2)  # 850 MB

# After optimization
df = pd.read_csv('large.csv',
                 dtype={'id': 'int32',
                        'category': 'category',
                        'value': 'float32'},
                 usecols=['id', 'category', 'value', 'date'],
                 parse_dates=['date'])
print(df.memory_usage(deep=True).sum() / 1024**2)  # 285 MB (66% reduction!)
```

---

## 7. Common Error Handling

### Error Resolution Guide

| Error | Cause | Solution |
|---|---|---|
| **UnicodeDecodeError** | Wrong encoding | Try `encoding='latin-1'` or `'cp1252'` |
| **ParserError** | Malformed rows | Use `error_bad_lines=False` |
| **FileNotFoundError** | Wrong path | Check file location |
| **MemoryError** | File too large | Use `chunksize` or load fewer columns |
| **DtypeWarning** | Mixed types | Specify `dtype` or use `low_memory=False` |

---

> **"Load Smart, Analyze Fast"**
> Always check `df.info()` for insights