from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import keras_tuner as kt
import pandas_ta as ta
import shutil
import io

app = Flask(__name__)

def run_model_for_ticker(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol.upper())
        hist = ticker.history(period="5y", interval="1d")

        # Technical indicators
        hist["RSI"] = ta.rsi(hist["Close"], length=14)
        hist["MACD"] = ta.macd(hist["Close"]).iloc[:, 0]
        hist["SMA_20"] = hist["Close"].rolling(20).mean()

        hist['future_return'] = (hist['Close'].shift(-7) - hist['Close']) / hist['Close']
        hist['label'] = (hist['future_return'] > 0.05).astype(int)

        df = hist.drop("future_return", axis=1).dropna()
        y = df.label.values
        X = df.drop(columns="label").values

        if y.sum() < 10:  # Not enough positive samples
            return "Not enough buy signals to train a meaningful model."

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.5)
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)

        # Compute class weights
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(zip(np.unique(y_train), weights))

        # Reset tuner
        shutil.rmtree('untitled_project', ignore_errors=True)

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

        tuner = kt.Hyperband(
            create_model,
            objective="val_accuracy",
            max_epochs=10,
            hyperband_iterations=1
        )

        tuner.search(X_train_scaled, y_train, epochs=10,
                     validation_data=(X_test_scaled, y_test),
                     class_weight=class_weights, verbose=0)

        best_model = tuner.get_best_models(1)[0]
        best_model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.2, class_weight=class_weights, verbose=0)

        y_pred_probs = best_model.predict(X_test_scaled)
        y_pred_classes = (y_pred_probs > 0.5).astype(int)

        report = classification_report(y_test, y_pred_classes, digits=3)
        return report

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        ticker = request.form.get("ticker")
        result = run_model_for_ticker(ticker)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)