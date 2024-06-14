import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

def preprocess_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y')
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day
    df['WITHDRAWAL_AMT'] = df[' WITHDRAWAL AMT '].replace({',': ''}, regex=True).astype(float)
    df['DEPOSIT_AMT'] = df[' DEPOSIT AMT '].replace({',': ''}, regex=True).astype(float)
    return df[['MONTH', 'DAY', 'WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'LOCATION', 'IP ADDRESS']]

def encode_categorical(df, cat_columns, categories):
    df_encoded = pd.get_dummies(df, columns=cat_columns)
    # Ensure all categories are present in encoded columns
    for col, cats in zip(cat_columns, categories):
        for cat in cats:
            if f"{col}_{cat}" not in df_encoded.columns:
                df_encoded[f"{col}_{cat}"] = 0
    return df_encoded

def predict_anomalies(model, data, scaler):
    # Get feature names seen during fit
    feature_names = scaler.get_feature_names_out()
    # Ensure input features match those seen during fit
    data = data.reindex(columns=feature_names, fill_value=0)
    data_scaled = scaler.transform(data)
    reconstructions = model.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
    return mse

# Load pre-trained autoencoder model
model_path = "fraud_detection_model_final.keras"
model = load_model(model_path)

# Load scaler
scaler_path = "scaler.pkl"
scaler = joblib.load(scaler_path)

# Load new data
new_data_path = "dataset.csv"
new_data = pd.read_csv(new_data_path)

# Preprocess new data
new_data_processed = preprocess_data(new_data)

# Encode categorical features
cat_columns = ['LOCATION', 'IP ADDRESS']
# Define categories for each categorical column
categories = [
    new_data_processed['LOCATION'].unique(),
    new_data_processed['IP ADDRESS'].unique()
]
new_data_encoded = encode_categorical(new_data_processed, cat_columns, categories)

# Predict anomalies
anomaly_scores = predict_anomalies(model, new_data_encoded, scaler)

# Output anomaly scores
for i, score in enumerate(anomaly_scores):
    print(f"Transaction {i + 1}: Anomaly Score - {score:.4f}")
