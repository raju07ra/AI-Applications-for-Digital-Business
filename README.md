# Customer Sentiment Analysis System (PoC)

**Module:** B198c7 AI Applications for Digital Business  
**Assessment:** Individual Project (Reassessment)  
**Submission Date:** January 8, 2026

## Project Overview
This repository contains the source code for my AI application designed for *ShopStream Retail*. The goal of this project was to automate the process of reading and tagging customer reviews.

Currently, the client processes reviews manually, which is slow and expensive. My solution uses Natural Language Processing (NLP) to automatically classify reviews as **Positive** or **Negative** with high accuracy.

## How It Works
The system is built in Python. It follows a standard Machine Learning pipeline:
1.  **Ingestion:** Loads the IMDb dataset (used as a proxy for customer feedback).
2.  **Preprocessing:** Cleans the text by removing HTML tags, punctuation, and stop words, then applies Porter Stemming.
3.  **Vectorization:** Converts text to numbers using **TF-IDF** (Term Frequency-Inverse Document Frequency).
4.  **Model:** Uses **Logistic Regression** to predict the sentiment.
5.  **Evaluation:** Calculates accuracy and generates a confusion matrix.

## Repository Structure
* `main.py`: The main script. Run this to train the model and see results.
* `requirements.txt`: List of Python libraries needed to run the code.
* `README.md`: This documentation file.
* `*.png`: (Generated after running code) Images for the report, including the Confusion Matrix and Feature Importance graph.

## How to Run This Project

### 1. Prerequisites
You need Python installed (version 3.8 or higher is recommended). You also need `pip` to install the required libraries.

### 2. Installation
Open your terminal or command prompt in this folder and run:
```bash
pip install -r requirements.txt

python main.py
