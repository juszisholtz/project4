import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import pandas_ta as ta
import numpy as np
import shutil

# Fetch data
ticker = yf.Ticker("GOOGL")
hist = ticker.history(period="10y", interval="1d")
hist["RSI"] = ta.rsi(hist["Close"], length=14)
hist["MACD"] = ta.macd(hist["Close"]).iloc[:, 0]
hist["SMA_20"] = hist["Close"].rolling(20).mean()

# Label
hist['future_return'] = (hist['Close'].shift(-7) - hist['Close']) / hist['Close']
hist['label'] = (hist['future_return'] > 0.05).astype(int)
clean_hist = hist.drop("future_return", axis=1).dropna()

y = clean_hist.label.values
X = clean_hist.drop(columns="label").values

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.5
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Clear old tuner
shutil.rmtree('untitled_project', ignore_errors=True)

# Build model function
def create_model(hp):
    model = tf.keras.models.Sequential()
    activation = hp.Choice('activation', ['relu','tanh','sigmoid'])
    model.add(tf.keras.layers.Dense(
        hp.Int('first_units',1,10,2), activation=activation,
        input_shape=(X_train_scaled.shape[1],)
    ))
    for i in range(hp.Int('num_layers',1,6)):
        model.add(tf.keras.layers.Dense(
            hp.Int(f'units_{i}',1,10,2),
            activation=activation
        ))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile("binary_crossentropy","adam",["accuracy"])
    return model

# Initialize tuner
import keras_tuner as kt
tuner = kt.Hyperband(
    create_model, objective="val_accuracy",
    max_epochs=20, hyperband_iterations=2
)

# Run hyperparameter search with tqdm callback
tuner.search(
    X_train_scaled, y_train,
    epochs=20,
    validation_data=(X_test_scaled, y_test),
    callbacks=[TqdmCallback(verbose=1)]
)

# Get best and retrain with progress bar
best_model = tuner.get_best_models(1)[0]
best_model.fit(
    X_train_scaled, y_train,
    epochs=20,
    validation_split=0.2,
    callbacks=[TqdmCallback(verbose=1)],
    verbose=0
)

# Evaluate
loss, acc = best_model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Report
y_pred_probs = best_model.predict(X_test_scaled)
threshold = np.percentile(y_pred_probs, 90)
y_pred = (y_pred_probs > threshold).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

