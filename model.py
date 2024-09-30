import numpy as np
import librosa
import joblib

# Define a function to extract MFCC features
def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.concatenate((np.mean(mfcc.T, axis=0), np.mean(mfcc_delta.T, axis=0), np.mean(mfcc_delta2.T, axis=0)))
    return mfcc_combined

# Load models and transformers
def load_models():
    nb_model = joblib.load('naive_bayes_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    return nb_model, svm_model, scaler, pca

# Predict function: Extract features, scale them, apply PCA, and predict using models
def predict(file_path):
    # Step 1: Load the audio file
    y, sr = librosa.load(file_path)

    # Step 2: Extract MFCC features
    features = extract_mfcc(y, sr)

    # Load models, scaler, and PCA
    nb_model, svm_model, scaler, pca = load_models()

    # Ensure the features have the same number of dimensions expected by the scaler
    if features.shape[0] != scaler.mean_.shape[0]:
        raise ValueError(f"Input features have {features.shape[0]} dimensions, "
                         f"but the scaler expects {scaler.mean_.shape[0]} dimensions.")

    # Step 3: Scale features
    features_scaled = scaler.transform([features])

    # Step 4: Apply PCA
    features_pca = pca.transform(features_scaled)

    # Step 5: Make predictions using Naive Bayes and SVM models
    nb_pred = nb_model.predict(features_pca)
    svm_pred = svm_model.predict(features_pca)

    return nb_pred[0], svm_pred[0]
