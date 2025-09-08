import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
import joblib

# -------------------------
# Load dataset
# -------------------------
file_path = r"D:\ML-civil\Copy of Biomedical waste ash dataset 600(1).xlsx"
df = pd.read_excel(file_path)

# Features and targets
X = df[['Cement(kg/m3)', 'Biomedical waste ash(kg/m3)',
        'Fine aggregate(kg/m3)', 'Coarse aggregate(kg/m3)']]

y = df[['Compressive strength (28 days)(MPa)',
        'Tensile strength(28 days)(MPa)',
        'Flexural strength(28 days)(MPa)']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------
# ANN Model
# -------------------------
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y.shape[1])   # 3 outputs
])

# ✅ FIX: use MeanSquaredError() instead of 'mse'
nn_model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

# Train
history = nn_model.fit(X_train_scaled, y_train, 
                       validation_data=(X_test_scaled, y_test),
                       epochs=100, batch_size=16, verbose=1)

# Save scaler
joblib.dump(scaler, "ann_scaler.pkl")

# Save ANN model (H5 for Keras + wrapper for sklearn-like usage)
nn_model.save("ann_model.h5")
joblib.dump({"scaler": scaler, "model_path": "ann_model.h5"}, "ann_model.pkl")

print("✅ ANN model and scaler saved successfully!")

