# Automatic-Review-Rating-System

## Project Overview
In today's digital era, customer reviews are a powerful source of feedback for businesses. Manually analyzing thousands of reviews to understand customer sentiment and assign a suitable rating is not only time-consuming but also subjective.

This project aims to build an Automated Review Rating System that can predict the star rating (1 to 5) of a product based solely on its review text. Using Natural Language Processing (NLP) techniques and machine learning, the system learns the patterns and sentiment behind review texts and maps them to the most probable rating class.
The project includes:

- Cleaning and preprocessing textual review data

- Creating balanced and imbalanced datasets for initial training and prototyping.

- Converting text to numerical features using TF-IDF

- Training a classifier (Logistic Regression) to predict ratings

- Evaluating performance using accuracy, precision, recall, and F1-score

## Data Collection 
For this project, we utilized a publicly available dataset sourced from Kaggle, titled “Cell Phones and Accessories - Amazon Reviews”. The dataset contains detailed customer reviews specifically for mobile phones and related accessories listed on Amazon.

Each row in the dataset represents a single user review and includes various metadata along with the actual review text and rating. This data is ideal for developing a review rating prediction system, as it provides both the natural language input (reviewText) and the corresponding target label (Rating).

### Dataset Details:
Source: Kaggle - Amazon Reviews: Cell Phones and Accessories

Domain: E-commerce / Amazon / Mobile Accessories

Size: ~190,000 reviews

Format: json (later converted to CSV(Comma-Separated Values))

[View Dataset](https://github.com/hannafarsin/automatic-review-rating-system/blob/main/data/kaggle/reviews_output.csv)

##  Data Preprocessing

The dataset was cleaned and preprocessed to improve the quality and usability of review text for machine learning:

- Removed exact duplicates and conflicting reviews (same text, different ratings)
- Converted all text to lowercase
- Removed punctuation and extra whitespace
- Removed stopwords using spaCy
- Applied lemmatization to normalize words to their base form
- Calculated review length (in words)

## Data Visualization

Several visualizations were used to explore and understand the data:

- Distribution of review ratings (bar chart and pie chart)
- Review length histogram

## Dataset Balancing

To avoid bias in the model due to class imbalance, we created a **balanced dataset** with **2,000 reviews per rating class** (from 1 to 5 stars). This ensured that the classifier was trained equally across all categories.

The original dataset after cleaning(180,000+ reviews) was filtered and sampled 


[View Balanced Dataset](https://github.com/hannafarsin/automatic-review-rating-system/blob/main/data/kaggle/balanced_reviews_dataset3.csv)


## Creating an Imbalanced Dataset

To simulate real-world scenarios where user ratings are not evenly distributed, we created an **imbalanced dataset**. This helps evaluate how a model trained on skewed data performs compared to a balanced one.

### Steps
- After cleaning and preprocessing the data, we retained the **original class distribution** without applying any resampling techniques.
- This dataset had significantly more **4-star and 5-star reviews**, while **1-star and 2-star reviews** were underrepresented.

### Motivation
- To train a second model (**Model_B**) on **naturally imbalanced data**.
- To compare its performance with **Model_A** (trained on balanced data) and observe the effects of training distribution on model generalization.

[View imbalanced dataset](https://github.com/hannafarsin/automatic-review-rating-system/blob/main/data/kaggle/imbalanced_reviews_dataset3.csv)

---

## Model Training

We trained two separate models using **Logistic Regression** — a reliable and interpretable baseline algorithm for text classification.
Common Training Steps:

-**Text Cleaning**: Lowercasing, punctuation and extra whitespace removed.

-**Stopword Removal**: Used spaCy to eliminate common non-informative words.

-**Lemmatization**: Reduced words to their base forms using spaCy.

-**Train-Test Split**: 80% training, 20% testing using stratified sampling.

-**Text Vectorization**: Applied TF-IDF (Term Frequency–Inverse Document Frequency)max_features=5000

Fitted on training set and transformed both train/test sets.

Model: Trained **Logistic Regression** (solver='liblinear') on TF-IDF vectors.

Serialization: Saved models and vectorizers using pickle for later use in the Streamlit app.

###  Process
- **Data Split**: 80% training and 20% testing (with stratification by class).
- **TF-IDF Vectorization**: Applied only on the training data to avoid data leakage.
- **Model Configuration**:
  - Classifier: `LogisticRegression`
  - Solver: `'liblinear'` (efficient for small and sparse datasets)
  - For **Model_A**: `class_weight='balanced'` was used to handle any residual class imbalance.
- **Serialization**: Both models and vectorizers were saved using `pickle` for deployment in the Streamlit interface.

### Model Variants

| Model    | Training Dataset                | Test Dataset                          |
|----------|----------------------------------|----------------------------------------|
| Model_A  | Balanced (Equal samples/class)   | Tested on both Balanced and Imbalanced |
| Model_B  | Imbalanced (Original distribution)| Tested on both Imbalanced and Balanced |

[View imbalanced Training Notebook](https://github.com/hannafarsin/automatic-review-rating-system/blob/main/notebook/balanced_training%206%20(1).ipynb)
[View imbalanced Training Notebook](https://github.com/hannafarsin/automatic-review-rating-system/blob/main/notebook/imbalanced%20training%20(1)%20(1).ipynb)

---

## Evaluation & Comparison

We used the following metrics to assess model performance:

- **Accuracy** – Overall percentage of correct predictions  
- **Precision** – Correct positive predictions / Total predicted positives  
- **Recall** – Correct positive predictions / Actual positives  
- **F1-score** – Harmonic mean of Precision and Recall  

### Summary of Results

| Model    | Train Set   | Test Set     | Accuracy | Observation                                                   |
|----------|-------------|--------------|----------|---------------------------------------------------------------|
| Model_A  | Balanced    | Balanced     | 0.44     | Best-case scenario; well-balanced across all classes          |
| Model_A  | Balanced    | Imbalanced   | 0.43     | Robust generalization; handles skewed data fairly             |
| Model_B  | Imbalanced  | Imbalanced   | 0.43     | Appears strong, but biased toward majority classes            |
| Model_B  | Imbalanced  | Balanced     | 0.44     | Improved performance; may generalize better than expected     |

###  Insight
Although both models achieved similar **accuracy**, **Model_A** showed more **consistent performance across class distributions**, while **Model_B** struggled with minority class predictions when tested on balanced data.


---

##  Streamlit Interface

To make the review rating system interactive and accessible, a **Streamlit web application** was developed.

### Features
- Simple UI to input any product review
- Real-time predictions from both models:
  - **Model_A** (Trained on Balanced Dataset)
  - **Model_B** (Trained on Imbalanced Dataset)
- **Side-by-side comparison** of predicted ratings

###  Deployment Stack

- **Frontend**: Streamlit components (e.g., text area, buttons, display)
- **Backend**: Pickle-loaded ML models and TF-IDF vectorizers

###  Key Files

| File/Folder | Description                               |
|-------------|-------------------------------------------|
| `app.py`    | Main Streamlit app script                 |
| `Models/`   | Folder containing pickled models and vectorizers |



---

## Deep Learning Overview

**Deep Learning** is a subfield of machine learning that uses neural networks with multiple layers to model complex patterns, especially in unstructured data like text, images, and audio.

###  When Deep Learning is Useful for Text Classification:
- A large dataset is available
- Context and word order are important
- Pretrained embeddings or models (like BERT) are used

---

## Deep Learning Models for Text Classification

Below are widely used deep learning architectures suitable for text classification:

### 1. RNN (Recurrent Neural Network)
- Handles sequential data
- Maintains memory through hidden states
- Struggles with long-range dependencies

### 2. LSTM (Long Short-Term Memory)
- Enhanced RNN with gating mechanisms
- Handles long-term dependencies
- Good for sequences where early words affect later ones

### 3. GRU (Gated Recurrent Unit)
- Similar to LSTM but more efficient (fewer gates)
- Faster with comparable performance

### 4. CNN (Convolutional Neural Network) for Text
- Captures local text features (e.g., n-grams)
- Effective for tasks like sentiment analysis

### 5. BERT (Bidirectional Encoder Representations from Transformers)
- Transformer-based, pre-trained on large corpora
- Captures bidirectional context
- Achieves state-of-the-art results in most NLP tasks

### 6. DistilBERT
- Lightweight version of BERT
- Retains ~95% of performance
- Faster and more efficient for real-time applications




