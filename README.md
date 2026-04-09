# Customer Sentiment Analysis System
## B198R AI Applications for Digital Business — Reassessment

**Student:** Rajkumar Labana  
**Client:** ShopStream Retail (Consumer Electronics)  
**Submission Date:** April 9, 2026  

## Project Overview
This repository contains the reassessment implementation of an AI-driven 
Sentiment Analysis System for ShopStream Retail. The system automatically 
classifies customer reviews as Positive or Negative using a TF-IDF and 
Logistic Regression pipeline.

## Reassessment Improvements
- Data cleaning audit (nulls, duplicates, row counts logged)
- Sample size stability test (5k, 10k, 15k rows compared)
- Bigrams implemented via ngram_range=(1,2)
- Negation tagging (NOT_ prefix preserves polarity context)

## Files
- `main.py` — Full pipeline: loading, cleaning, preprocessing, training, evaluation
- `requirements.txt` — All required libraries
- `README.md` — This file

## How to Run
```bash
pip install -r requirements.txt
python main.py
```

## Requirements
- Python 3.10+
- datasets
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- nltk
