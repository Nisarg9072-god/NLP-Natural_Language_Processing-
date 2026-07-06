# 🧠 NLP — Emotion Detection from Text

A Natural Language Processing project that classifies text into **6 emotions** (sadness, anger, love, surprise, fear, joy) using classic ML techniques including Bag-of-Words and TF-IDF with a Naive Bayes classifier.

---

## 📁 Project Structure

```
NLP(Natural Language processing)/
│
├── main.py          # Main pipeline: preprocessing → vectorization → classification
├── count.py         # TF-IDF vectorization demo
├── train.txt        # Dataset (text;emotion format)
├── requirements.txt # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd "NLP(Natural Language processing)"
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python main.py
```

---

## 🔄 Pipeline

```
Raw Text (train.txt)
      │
      ▼
  Lowercasing
      │
      ▼
  Remove Punctuation
      │
      ▼
  Remove Digits
      │
      ▼
  Remove @user / URL
      │
      ▼
  Stop Word Removal (NLTK)
      │
      ├──► BoW Vectorizer  ──► Multinomial Naive Bayes ──► Accuracy / Report
      │
      └──► TF-IDF Vectorizer ──► Multinomial Naive Bayes ──► Accuracy / Report
```

---

## 📊 Results

| Vectorizer | Accuracy |
|------------|----------|
| Bag-of-Words (BoW) | **76.5%** |
| TF-IDF | **66.1%** |

BoW outperforms TF-IDF here because Naive Bayes works well with raw term counts.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data loading & manipulation |
| `nltk` | Tokenization & stop word removal |
| `scikit-learn` | Vectorization & Naive Bayes classifier |
| `matplotlib` / `seaborn` | Visualization |

---

## 📝 Dataset

- **File:** `train.txt`
- **Format:** `text;emotion` (semicolon-separated, no header)
- **Size:** 16,000 samples
- **Classes:** sadness, anger, love, surprise, fear, joy

---

## 📌 Notes

- NLTK data (`punkt_tab`, `stopwords`) is downloaded automatically on the first run.
- The `from enum import unique` import at the top of `main.py` is unused and can be safely removed.
