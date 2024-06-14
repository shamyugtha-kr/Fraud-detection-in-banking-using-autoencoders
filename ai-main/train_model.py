import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib

# Load data
file_path = "train_data.csv"
df = pd.read_csv(file_path)
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y')
df = df.sort_values(by='DATE')

# Feature engineering
df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
df['MONTH'] = df['DATE'].dt.month
df['DAY'] = df['DATE'].dt.day
df['WITHDRAWAL_AMT'] = df['WITHDRAWAL AMT'].replace({',': ''}, regex=True).astype(float)

# Drop non-numeric columns and NaN values
df_numeric = df[['MONTH', 'DAY', 'WITHDRAWAL_AMT']]
df_numeric = df_numeric.dropna()

# Scaling features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Define and train autoencoder model
input_dim = df_scaled.shape[1]
encoding_dim = 32  # Adjust encoding dimension as needed

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(df_scaled, df_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Save autoencoder model and scaler
model_path = "fraud_detection_model_final.keras"
scaler_path = "scaler.pkl"
autoencoder.save(model_path)
joblib.dump(scaler, scaler_path)
