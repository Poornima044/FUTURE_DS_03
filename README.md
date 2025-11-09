# 📘 College Event Feedback Analysis – README.md

## 🎓 Project Overview
This project analyzes college event feedback collected from students through Google Forms. Using Python, pandas, and sentiment analysis tools, it extracts meaningful insights to improve future campus events.

---

## 🎯 Objectives
- Clean and preprocess feedback data from CSV.
- Analyze numerical ratings (1–5 scale).
- Perform sentiment analysis on text feedback using **TextBlob** and **VADER**.
- Create visualizations to show trends and satisfaction levels.
- Suggest recommendations for improvement.

---

## 🧠 Skills Demonstrated
- Data Cleaning & Preparation (pandas)
- Exploratory Data Analysis (EDA)
- Sentiment Analysis (TextBlob / VADER)
- Data Visualization (Matplotlib / Seaborn / WordCloud)
- Report Generation

---

## 🛠️ Tools & Libraries
| Tool/Library | Purpose |
|---------------|----------|
| **Google Colab** | Run code online without setup |
| **pandas** | Data manipulation |
| **seaborn / matplotlib** | Visualization |
| **TextBlob / VADER** | Sentiment analysis |
| **WordCloud** | Text visualization |

---

## 📂 Dataset
**File:** `student_feedback.csv`

**Columns:**
| Column | Description |
|---------|-------------|
| `Timestamp` | When feedback was given |
| `Event_Name` | Name of the college event |
| `Department` | Organizing department |
| `Rating` | Student rating (1–5) |
| `Feedback` | Text feedback / comments |

You can collect this data via Google Forms and export it as CSV.

---

## 🧹 Step 1: Data Cleaning
- Load CSV file.
- Remove duplicates.
- Standardize column names.
- Handle missing or invalid ratings.

```python
import pandas as pd

df = pd.read_csv('student_feedback.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df.drop_duplicates(inplace=True)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
```

---

## 📊 Step 2: Rating Analysis
Visualize rating distribution and find top-rated events.

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
sns.histplot(df['rating'], bins=5, kde=True)
plt.title('Distribution of Ratings')
plt.show()
```

---

## 💬 Step 3: Sentiment Analysis
Perform polarity-based sentiment classification.

```python
from textblob import TextBlob

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['feedback'].apply(get_sentiment)
```

---

## ☁️ Step 4: Word Cloud Visualization

```python
from wordcloud import WordCloud

all_text = ' '.join(df['feedback'].dropna())
wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
```

---

## 📈 Step 5: Insights & Visuals
- Top events by rating.
- Sentiment distribution by department.
- Correlation between ratings and sentiment.

```python
sns.countplot(data=df, x='department', hue='sentiment')
plt.title('Sentiment by Department')
plt.show()
```

---

## 💡 Key Insights
- **Top Events:** Hackathon, Cultural Fest, AI Workshop.
- **Positive Feedback:** Speakers, learning experience, coordination.
- **Improvement Areas:** Food, time management, event length.

---

## 🧾 Recommendations
- Enhance logistics and scheduling.
- Maintain engaging activities like hackathons.
- Collect more detailed feedback after each event.

---

## 📁 Deliverables
✅ Google Colab Notebook / Jupyter Notebook  
✅ Graphs & Word Clouds  
✅ README File (this document)  
✅ Insights Summary (PDF or Markdown)

---

## 🏁 Conclusion
This project demonstrates how **data-driven insights** can improve campus event planning. By combining ratings and sentiment analysis, colleges can make informed decisions to enhance student experiences.
