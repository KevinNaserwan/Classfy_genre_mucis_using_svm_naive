import streamlit as st
import librosa
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load models and transformers
def load_models():
    nb_model = joblib.load('naive_bayes_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    return nb_model, svm_model, scaler, pca

# Define a function to extract MFCC features
def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.concatenate((np.mean(mfcc.T, axis=0), np.mean(mfcc_delta.T, axis=0), np.mean(mfcc_delta2.T, axis=0)))
    return mfcc_combined

# Predict function using loaded models
def predict(file_path, model_choice):
    y, sr = librosa.load(file_path)
    features = extract_mfcc(y, sr)
    nb_model, svm_model, scaler, pca = load_models()
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)

    # Predict based on model choice
    if model_choice == "Naive Bayes":
        return nb_model.predict(features_pca)[0], None
    elif model_choice == "SVM":
        return None, svm_model.predict(features_pca)[0]
    else:
        return nb_model.predict(features_pca)[0], svm_model.predict(features_pca)[0]

# Main app structure
st.title("Audio Genre Classification")

# Sidebar for page navigation
page = st.sidebar.selectbox("Choose a page:", ["Upload Audio", "Choose Model", "Results"])

if page == "Upload Audio":
    st.header("Upload Audio File")
    st.write("Please upload a WAV file to classify its genre.")
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['uploaded_file'] = "temp.wav"
        st.success("File uploaded successfully!")

elif page == "Choose Model":
    st.header("Select Classification Model")
    if 'uploaded_file' not in st.session_state:
        st.warning("Please upload a file first.")
    else:
        model_choice = st.radio("Choose a model for classification:", ("Naive Bayes", "SVM", "Both"))
        st.session_state['model_choice'] = model_choice
        if st.button("Submit"):
            nb_pred, svm_pred = predict(st.session_state['uploaded_file'], model_choice)
            st.session_state['nb_pred'] = nb_pred
            st.session_state['svm_pred'] = svm_pred
            st.success("Model selection successful, proceed to results.")

elif page == "Results":
    st.header("Classification Results")
    if 'nb_pred' in st.session_state or 'svm_pred' in st.session_state:
        if st.session_state['nb_pred'] is not None:
            st.write(f"**Naive Bayes Prediction:** {st.session_state['nb_pred']}")
        if st.session_state['svm_pred'] is not None:
            st.write(f"**SVM Prediction:** {st.session_state['svm_pred']}")

        # # Example performance comparison data (dummy data for the table)
        # comparison_data = {
        #     'Model': ['Naive Bayes', 'SVM'],
        #     'Accuracy': [0.85, 0.90],
        #     'Precision': [0.83, 0.88],
        #     'Recall': [0.82, 0.87]
        # }

        # # Convert to dataframe
        # comparison_df = pd.DataFrame(comparison_data)

        # # Display the comparison table
        # st.write("### Model Performance Comparison")
        # st.table(comparison_df)
    else:
        st.warning("No results to display. Please upload and classify an audio file first.")