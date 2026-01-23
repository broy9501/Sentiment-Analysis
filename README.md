# 🧠 Sentiment Analysis NLP Project

## 📌 Overview
This project is a **Sentiment Analysis system built using Natural Language Processing (NLP)**.  
It analyses user-provided text and identifies the **emotional tone and sentiment** behind it.  

The goal of this project is to move beyond simple keyword matching and instead use **machine learning–based text representations** to understand how a user feels, making it suitable for real-world applications such as emotional AI assistants, mental wellbeing tools, and personalised systems.

---

## 🎯 Project Objectives
- Analyse free-form user text
- Detect **sentiment polarity** (e.g. positive, negative, neutral)
- Identify **emotional signals** within text
- Demonstrate applied NLP using modern vectorisation techniques
- Lay the groundwork for emotion-aware AI systems

---

## 🧠 How It Works (Pipeline)
1. **User Input**
   - The user provides a sentence or paragraph of text.

2. **Text Preprocessing**
   - Text is cleaned and prepared for analysis.

3. **Vectorisation**
   - Text is converted into numerical form using NLP techniques such as:
     - TF-IDF

4. **Similarity & Analysis**
   - Cosine similarity is used to compare user input against known sentiment or emotion patterns.

5. **Sentiment & Emotion Detection**
   - The system outputs:
     - Positive, Neutral, Negative

---

## 🛠️ Technologies Used
- **Python**
- **scikit-learn**
- **SentenceTransformers**
- **Pandas**
- **Cosine Similarity**
- **TF-IDF Vectorisation**

---

## 📥 Example Input
```text
That’s done and I feel lighter now.

