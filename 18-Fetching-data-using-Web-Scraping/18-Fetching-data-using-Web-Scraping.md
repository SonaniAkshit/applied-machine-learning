# Fetching Data Using Web Scraping

---

## 1. Intuition

Web scraping means:

> Extracting structured data from websites that were built for humans, not for machines.

Websites show data in HTML format.
You scrape when:

* No API is available
* API is expensive
* You need public data at scale
* You are building datasets for ML

Example:

* E-commerce price tracking
* News sentiment analysis
* Job market trend analysis
* Real estate price prediction

Scraping is not hacking.

It is:

* Sending HTTP requests
* Parsing HTML
* Extracting useful parts
* Converting to structured format (CSV / JSON / DB)

---

## 2. When Should You Use Web Scraping?

Be honest here. Don’t scrape blindly.

### ✅ Use scraping when:

* Website has public data
* No official API exists
* Data is updated frequently
* You need automation

### ❌ Don’t use scraping when:

* API exists (always prefer API)
* Website prohibits scraping in robots.txt
* Data is sensitive
* Legal issues exist

Industry rule:

> API > Database > Scraping

Scraping is last resort.

---

## 3. How Websites Actually Work (You Must Understand This)

When you open a website:

1. Browser sends request
2. Server sends HTML
3. Browser renders page

Basic scraping replicates step 1 and 2.

If you don’t understand HTTP requests, you’re guessing.

---

## 4. Types of Web Scraping

### 1️⃣ Static Websites

* HTML already contains data
* Easy to scrape

### 2️⃣ Dynamic Websites

* Data loads using JavaScript
* Requires automation tools (Selenium / Playwright)

If you try `requests` on dynamic site and it fails, don’t panic.
It means data loads via JS.

---

## 5. Tools Used in Python

* `requests` → Send HTTP request
* `BeautifulSoup` → Parse HTML
* `lxml` → Fast parser
* `Selenium` → Browser automation
* `Playwright` → Modern automation
* `Scrapy` → Large-scale scraping framework

For ML beginners:
Start with:

* requests
* BeautifulSoup

---

# Step-by-Step Example (Static Website)

We’ll use a simple demo site:

🔗 [http://quotes.toscrape.com](http://quotes.toscrape.com)
(This site is made for learning scraping.)

---

## Step 1: Install Libraries

```bash
pip install requests beautifulsoup4
```

---

## Step 2: Send HTTP Request

```python
import requests

url = "http://quotes.toscrape.com"
response = requests.get(url)

print(response.status_code)
```

If status code = 200 → success
If 403 → blocked
If 404 → page not found

Industry mindset:
Always check status code.

---

## Step 3: Parse HTML

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(response.text, "html.parser")

print(soup.title.text)
```

Now you have full HTML tree.

---

## Step 4: Inspect Website (VERY IMPORTANT)

Right-click → Inspect
Find the HTML structure.

For quotes site:

Each quote is inside:

```html
<div class="quote">
```

So we extract all such divs.

---

## Step 5: Extract Data

```python
quotes = soup.find_all("div", class_="quote")

for quote in quotes:
    text = quote.find("span", class_="text").text
    author = quote.find("small", class_="author").text
    
    print(text)
    print(author)
    print("-" * 50)
```

Now you’re extracting structured data.

---

## Step 6: Convert to Structured Format

```python
import pandas as pd

data = []

for quote in quotes:
    text = quote.find("span", class_="text").text
    author = quote.find("small", class_="author").text
    
    data.append({
        "quote": text,
        "author": author
    })

df = pd.DataFrame(data)
df.to_csv("quotes.csv", index=False)
```

Now you have dataset.

This is how ML datasets are built.

---

# Dynamic Website Example (Concept Only)

If site loads content using JavaScript:

`requests` won’t see data.

Then use Selenium:

```bash
pip install selenium
```

Basic idea:

```python
from selenium import webdriver

driver = webdriver.Chrome()
driver.get("https://example.com")

html = driver.page_source
```

Then parse with BeautifulSoup.

But remember:

Selenium is heavy.
Don’t use it unless necessary.

---

# Real ML Use Case Example

Suppose you want to build:

📊 Real Estate Price Prediction Model

Steps:

1. Scrape property listings
2. Extract:

   * price
   * location
   * sqft
   * bedrooms
3. Clean data
4. Train regression model

Without scraping, no dataset.

---

# Production-Level Thinking (This Is Where Most Beginners Fail)

Scraping once is easy.

Scaling is hard.

Think like engineer:

* What if website layout changes?
* What if IP gets blocked?
* How to rotate proxies?
* How to schedule scraping daily?
* How to avoid duplicates?
* How to monitor failures?

In production:

* Use retry logic
* Use delay between requests
* Respect rate limits
* Log errors
* Use cron jobs or Airflow

If you ignore this, your pipeline breaks.

---

# Common Beginner Mistakes

1. Scraping without checking robots.txt
2. No delay between requests
3. No error handling
4. Hardcoding HTML structure blindly
5. Not handling pagination
6. Ignoring encoding issues
7. Not cleaning data

---

# When Web Scraping Fails

* JavaScript-heavy sites
* CAPTCHA
* Login required sites
* Anti-bot protection

Then you need:

* APIs
* Browser automation
* Scraping services
* Reverse engineering network calls

---

# Industry Reality Check

Let me be direct:

If your ML project depends entirely on scraping random websites without thinking about:

* Legal compliance
* Scalability
* Data quality
* Maintenance

It’s not production-level ML.

It’s a toy project.

---

# Your Turn (No Skipping)

Answer these:

1. What is the difference between static and dynamic scraping?
2. Why is API preferred over scraping?
3. If website layout changes, how does it affect your ML pipeline?
4. How would you avoid getting blocked?

Think like an engineer. Not a tutorial watcher.

When you answer, I’ll push you further.
