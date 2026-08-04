# NLP Pipeline Demo 🧠

An interactive Streamlit web app that demonstrates the core steps of a 
Natural Language Processing (NLP) pipeline in real-time.

## Features
- Live text statistics (character, word, and sentence count)
- Sentence & Word Tokenization (NLTK)
- Stopword Removal
- Part-of-Speech (POS) Tagging
- Lemmatization
- Named Entity Recognition (NER) using spaCy
- Token-level breakdown (Token, Lemma, POS, Dependency) in a clean table

## Tech Stack
- Python
- Streamlit — for the interactive UI
- NLTK — tokenization, stopwords, POS tagging, lemmatization
- spaCy — Named Entity Recognition
- Pandas — tabular token analysis

## How to Run
```bash
pip install streamlit nltk spacy pandas
python -m spacy download en_core_web_sm
streamlit run Practical1.py
```

## What it Demonstrates
This project shows how raw text goes through each stage of NLP 
preprocessing — from splitting sentences down to identifying named 
entities — making it a good learning/demo tool for understanding 
how NLP pipelines work under the hood.
