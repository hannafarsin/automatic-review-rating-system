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

## Text Vectorization

To transform textual review data into a machine-readable format, we used the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique with the following settings:

- Maximum features: 5000
- Trained only on the training set
- Transformed both training and testing text

TF-IDF allowed us to extract the most important words across the corpus without giving high weight to common stopwords.



[View Balanced Training Notebook](https://github.com/hannafarsin/automatic-review-rating-system/blob/main/notebook/balanced_training%20(2)%20(1).ipynb)






