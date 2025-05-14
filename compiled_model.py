import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import sklearn as skl
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pandas_ta as ta
import keras_tuner as kt
import shutil

# Get historical stock data
ticker = yf.Ticker("GOOGL")
hist = ticker.history(period="10y", interval="1d")

# Add technical indicators
hist["RSI"] = ta.rsi(hist["Close"], length=14)
hist["MACD"] = ta.macd(hist["Close"]).iloc[:, 0]
hist["SMA_20"] = hist["Close"].rolling(20).mean()

# Create label
hist['future_return'] = (hist['Close'].shift(-7) - hist['Close']) / hist['Close']
hist['label'] = (hist['future_return'] > 0.05).astype(int)

# Clean data
clean_hist = hist.drop("future_return", axis=1).dropna()

# Prepare features and target
y = clean_hist.label.values
X = clean_hist.drop(columns="label").values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.5)

# Scale features
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Reset tuner
shutil.rmtree('untitled_project', ignore_errors=True)

# Define model builder
def create_model(hp):
    model = tf.keras.models.Sequential()
    activation = hp.Choice('activation', ['relu', 'tanh', 'sigmoid'])

    model.add(tf.keras.layers.Dense(
        units=hp.Int('first_units', min_value=1, max_value=10, step=2),
        activation=activation,
        input_shape=(X_train_scaled.shape[1],)
    ))

    for i in range(hp.Int('num_layers', 1, 6)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=1, max_value=10, step=2),
            activation=activation
        ))

    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model

# Initialize tuner
tuner = kt.Hyperband(
    create_model,
    objective="val_accuracy",
    max_epochs=20,
    hyperband_iterations=2
)

# Search for best model
tuner.search(X_train_scaled, y_train, epochs=20,
             validation_data=(X_test_scaled, y_test))

# Retrieve and retrain best model
best_model = tuner.get_best_models(1)[0]

# Evaluate on test data
loss, accuracy = best_model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed classification report
y_pred_probs = best_model.predict(X_test_scaled)
y_pred_classes = (y_pred_probs > np.percentile(y_pred_probs, 90)).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))
