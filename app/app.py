import streamlit as st
import pickle

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer(path_model, path_vectorizer):
    model = pickle.load(open(path_model, 'rb'))
    vectorizer = pickle.load(open(path_vectorizer, 'rb'))
    return model, vectorizer

# Absolute or relative paths to your files
model_A, tfidf_A = load_model_and_vectorizer(
    'Models/model_A.pkl', 'Models/TfidfVectorizer_A.pkl')
model_B, tfidf_B = load_model_and_vectorizer(
    'Models/Model_B.pkl', 'Models/TfidfVectorizer_b.pkl')

# Streamlit UI
st.title("Review Rating Predictor")

user_input = st.text_area(" Enter your product review here:")

if st.button("Predict Ratings"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X_input_A = tfidf_A.transform([user_input])
        X_input_B = tfidf_B.transform([user_input])

        pred_A = model_A.predict(X_input_A)[0]
        pred_B = model_B.predict(X_input_B)[0]

        st.success(f"Model A (Balanced) Prediction: ⭐ {pred_A}")
        st.info(f"Model B (Imbalanced) Prediction: ⭐ {pred_B}")
