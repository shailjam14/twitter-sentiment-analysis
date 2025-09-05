Twitter Sentiment Analysis (LSTM + GloVe)

This project performs sentiment analysis on tweets using deep learning (LSTM) with pre-trained GloVe embeddings.
The goal is to classify tweets as positive or negative.

 Project Overview

Preprocesses raw tweets (tokenization, lemmatization, stopword handling).

Converts words into GloVe word vectors.

Trains a stacked LSTM model with dropout regularization.

Evaluates using Accuracy, AUC, and Classification Report.

 Features

✅ Clean text preprocessing (tokenization + lemmatization)
✅ Word embeddings with GloVe (Global Vectors for Word Representation)
✅ Deep learning model with LSTMs
✅ Model checkpointing to save best weights
✅ Performance evaluation with multiple metrics


Labels:

1 → Positive

0 → Negative

Dataset is not included in the repo (you can download from Kaggle - Twitter Sentiment Analysis
 or your own source).

🏗 Model Architecture

The deep learning pipeline uses:

Input layer (57×50 padded GloVe embeddings)

3 stacked LSTM layers (64 units each) with Dropout(0.2)

Flatten layer

Dense layer with sigmoid activation → Binary classification

 Results

Performance on test data:

Metric	Score
Accuracy	~85%
AUC	~0.90
Precision	~0.84
Recall	~0.86

(These numbers may vary depending on dataset splits and training runs)

You can also visualize:

Training vs Validation Loss

Training vs Validation Accuracy

How to Run
1Clone the repository
git clone https://github.com/shailjam14/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

 Install dependencies
pip install -r requirements.txt

 Download GloVe embeddings
wget http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip

 Add dataset

Place train.csv in the project root directory.

Train the model
python twitter_anaylsis.py

Tech Stack

Python

Pandas, NumPy

NLTK

TensorFlow / Keras

Scikit-learn

Matplotlib, Seaborn

 Future Work

🔹 Deploy model using Streamlit or Flask API

🔹 Use transformers (BERT, RoBERTa) for better accuracy

🔹 Perform hyperparameter tuning & cross-validation

🔹 Extend dataset for multi-class sentiment (positive, negative, neutral)
🔹 Perform hyperparameter tuning & cross-validation

🔹 Extend dataset for multi-class sentiment (positive, negative, neutral)
