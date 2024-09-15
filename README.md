
Veracity Vigilance: Leveraging Machine Learning for Fake News Detection

Project Overview

Veracity Vigilance is a machine learning project aimed at detecting fake news. This project utilizes Natural Language Processing (NLP) techniques and machine learning models to classify news articles as either real or fake. The primary goal is to build a reliable model that can differentiate between genuine and false news articles.

Table of Contents

1. Project Description
2. Dataset
3. Setup and Installation
4. Preprocessing
5. Model Training and Evaluation
6. Usage
7. Results
8. License
9. Acknowledgements

Project Description

This project involves the following key steps:
1. Data Collection: Gathered a dataset containing news articles with labels indicating whether the news is real or fake.
2. Preprocessing: Cleaned and transformed the text data to make it suitable for machine learning models.
3. Model Training: Trained a `Multinomial Naive Bayes` classifier on the processed text data.
4. Evaluation: Evaluated the model's performance using metrics such as accuracy, precision, recall, and F1-score.

Dataset

The dataset used in this project is a combination of two datasets:
- True News: Contains genuine news articles.
- Fake News: Contains fabricated news articles.

Each dataset was reduced to 4000 samples, resulting in a balanced dataset of 8000 samples (4000 real and 4000 fake news articles).

Setup and Installation

To set up this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kavitharavichandran3805/Fake_new_detection/
   
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Make sure the `requirements.txt` file includes:
   ```
   numpy
   pandas
   scikit-learn
   spacy
   nltk
   matplotlib
   seaborn
   ```

   Also, download the necessary NLTK and spaCy models:
   ```python
   import nltk
   nltk.download('punkt')
   ```

   ```bash
   python -m spacy download en_core_web_sm
   ```

Preprocessing

1. Text Lowercasing: Convert all text to lowercase.
2. Tokenization: Split text into individual words.
3. Remove Punctuation: Remove punctuation and special characters.
4. Vectorization: Transform text into numerical feature vectors using `TfidfVectorizer`.

Model Training and Evaluation

1. Split Data: Split the dataset into training and testing sets.
2. Train Model: Train a `Multinomial Naive Bayes` classifier.
3. Evaluate Model: Evaluate the model using accuracy, precision, recall, and F1-score metrics.

python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(dataset['text'])
y = dataset['category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


Usage

To use the trained model for predictions, load the model and vectorizer, and preprocess the new text data in the same way as done during training.

Results

The model achieved an accuracy of 93% on the test dataset. Further details about the model's performance can be found in the classification report.

 Acknowledgements

- The dataset was sourced from Kaggle.

