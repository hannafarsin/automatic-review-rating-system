from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

app = Flask(__name__)

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Load models
model_balanced = load_model("../Models/deep_balanced_model4.h5")
model_imbalanced = load_model("../Models/deep_imbalanced_model3.h5")

# Load tokenizers and max_len
with open("../Models/deep_tokenizer_balanced.pkl", "rb") as f:
    tokenizer_balanced = pickle.load(f)
with open("../Models/maxlen_balanced1.pkl", "rb") as f:
    maxlen_balanced = pickle.load(f)

with open("../Models/deep_tokenizer_imbalanced.pkl", "rb") as f:
    tokenizer_imbalanced = pickle.load(f)
with open("../Models/maxlen_imbalanced1.pkl", "rb") as f:
    maxlen_imbalanced = pickle.load(f)


def spacy_preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_balanced = None
    prediction_imbalanced = None
    review_text = ""
    warning = None  # Optional: display warning on empty input

    if request.method == "POST":
        review_text = request.form["review"].strip()

        if review_text == "":
            # Clear predictions if no input
            prediction_balanced = None
            prediction_imbalanced = None
            warning = "Please enter a review before submitting."
        else:
            cleaned_text = spacy_preprocess(review_text)

            # Tokenize and predict for balanced model
            seq_balanced = tokenizer_balanced.texts_to_sequences([cleaned_text])
            padded_balanced = pad_sequences(seq_balanced, maxlen=maxlen_balanced, padding="post", truncating="post")
            pred_balanced = model_balanced.predict(padded_balanced)
            prediction_balanced = np.argmax(pred_balanced) + 1

            # Tokenize and predict for imbalanced model
            seq_imbalanced = tokenizer_imbalanced.texts_to_sequences([cleaned_text])
            padded_imbalanced = pad_sequences(seq_imbalanced, maxlen=maxlen_imbalanced, padding="post", truncating="post")
            pred_imbalanced = model_imbalanced.predict(padded_imbalanced)
            prediction_imbalanced = np.argmax(pred_imbalanced) + 1

    return render_template(
        "index.html",
        prediction_balanced=prediction_balanced,
        prediction_imbalanced=prediction_imbalanced,
        review_text=review_text,
        warning=warning
    )


if __name__ == "__main__":
    app.run(debug=True)
