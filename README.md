# NLP-Flight-intent-classifier
 NLP Flight Intent Classifier
# ✈️ NLP Flight Intent Classifier

A Natural Language Processing web app that takes a customer's airline 
query and instantly classifies it into the correct department or intent 
category — powered by TF-IDF and Logistic Regression.

Built as part of a daily ML streak, following a documented failed attempt 
on synthetic data that proved data quality determines model performance.

## 🌐 Live Demo
[Add your Streamlit Cloud link here]

## 🎯 What It Does
- Customer types a natural language airline query
- Model classifies it into one of 9 intent categories
- Shows confidence score — flags low confidence predictions honestly
- Displays how the text was cleaned and classified

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| Algorithm | Logistic Regression + TF-IDF |
| Accuracy | 92% |
| Best class F1 | atis_flight_time — 1.00 |
| Weakest class F1 | atis_other — 0.56 |
| Training data | 4,977 real customer queries |

## 🧠 NLP Pipeline
- Lowercased, removed apostrophes, punctuation, numbers
- Removed 179 NLTK stopwords
- Lemmatized words to root form
- TF-IDF with bigrams (ngram_range=(1,2))
- class_weight='balanced' for class imbalance

## 📖 The Failure That Led Here
Same pipeline on synthetic data → 19% accuracy.
Same pipeline on real data → 92% accuracy.

> Data quality determines your model's ceiling.
> Algorithm selection is secondary.

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py

## 💡 Key Learnings
- Real vs synthetic data — same pipeline, 19% becomes 92%
- Bigrams capture context unigrams miss
- Confidence scores make models honest
- Save both the pipeline AND the label encoder for deployment

## 👩🏾‍💻 Author
Thelma — Mechatronics Engineering Student | Data Science & AI/ML
