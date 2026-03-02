# Fetching Data from an API and Creating a Pandas DataFrame

---

## 1. Intuition

An API (Application Programming Interface) is basically a **bridge between two systems**.

Instead of reading a file like:

```python
pd.read_csv("data.csv")
```

You send a **request** to a server:

> "Hey server, give me the data."

The server responds usually in **JSON format**.

Then you:

1. Receive JSON
2. Convert JSON → Python dictionary/list
3. Convert that → Pandas DataFrame

That’s the full flow.

---

## 2. What is an API (in practical ML terms)?

Most ML systems pull data from:

* Payment systems
* User databases
* Weather services
* Social media platforms
* Internal microservices

Example APIs:

* OpenWeather API (weather data)
* GitHub API (repo data)
* Twitter API (tweets)
* NASA Open APIs (space data)

They return structured JSON.

---

## 3. What Does API Data Look Like?

Typical response (JSON):

```json
[
  {
    "id": 1,
    "name": "Akshit",
    "age": 22
  },
  {
    "id": 2,
    "name": "Rahul",
    "age": 25
  }
]
```

This is basically:

* A list
* Of dictionaries

And that is perfect for pandas.

---

## 4. Step-by-Step Process (Industry Way)

### Step 1: Send Request

We use:

```python
import requests
```

Because pandas does NOT fetch APIs directly.

---

### Step 2: Get Response

```python
response = requests.get("API_URL")
```

Important check:

```python
response.status_code
```

If it’s not 200, your pipeline is broken.

In production:

* You NEVER assume API works.
* You handle failures.

---

### Step 3: Convert to JSON

```python
data = response.json()
```

Now `data` is a Python object.

---

### Step 4: Convert to DataFrame

```python
import pandas as pd

df = pd.DataFrame(data)
```

Done.

---

## 5. Real Example (Public API)

Example API:

```
https://jsonplaceholder.typicode.com/users
```

This returns fake user data.

Full clean code:

```python
import requests
import pandas as pd

url = "https://jsonplaceholder.typicode.com/users"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    print(df.head())
else:
    print("Failed to fetch data")
```

---

## 6. What If JSON is Nested?

Real APIs are messy.

Example:

```json
{
  "user": {
    "id": 1,
    "name": "Akshit"
  },
  "orders": [
    {"product": "Laptop", "price": 50000},
    {"product": "Mouse", "price": 1000}
  ]
}
```

This cannot directly become a clean table.

You may need:

```python
pd.json_normalize(data)
```

Or:

```python
pd.json_normalize(data, record_path="orders")
```

This is VERY common in real ML pipelines.

---

## 7. Common Beginner Mistakes

Let me be blunt:

### ❌ 1. Not checking status code

That’s careless.

### ❌ 2. Assuming JSON structure

Always inspect:

```python
print(data)
```

### ❌ 3. Ignoring nested fields

Most APIs are nested.

### ❌ 4. No error handling

In production, APIs:

* Timeout
* Return 500
* Return incomplete data

You must handle it.

---

## 8. Production-Level Thinking

Now think like an ML engineer.

Ask yourself:

### What can go wrong?

* API rate limits
* Authentication required (API keys)
* Data schema changes
* Missing fields
* Slow response

### How to handle?

* Retry logic
* Logging
* Validation checks
* Schema enforcement
* Monitoring

In real companies, API failures can break training pipelines.

---

## 9. When to Use API Data in ML?

Use APIs when:

* Data updates frequently
* Real-time prediction required
* Integrating external signals (weather, stock, user activity)

Example:
A food delivery model using:

* Weather API
* Traffic API
* User activity API

That’s industry-level thinking.

---

## 10. Tough Question for You

Answer this before moving ahead:

If API returns:

```json
{
  "data": [
    {"id": 1, "score": 10},
    {"id": 2, "score": 15}
  ]
}
```

Will `pd.DataFrame(data)` work correctly?

If not:

* Why?
* How would you fix it?

Think.
Don’t rush to code.

Reply with your reasoning.
