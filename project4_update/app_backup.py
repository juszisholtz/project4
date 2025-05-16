from flask import Flask, request, render_template
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import shutil
import pandas_ta as ta
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

def run_model_for_ticker(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol.upper())
        hist = ticker.history(period="10y", interval="1d")

        if hist.empty or len(hist) < 300:
            return "Not enough historical data to train the model.", "", "", ""

        # Add indicators
        hist["RSI"] = ta.rsi(hist["Close"], length=14)
        hist["MACD"] = ta.macd(hist["Close"]).iloc[:, 0]
        hist["SMA_20"] = hist["Close"].rolling(20).mean()
        hist['future_return'] = (hist['Close'].shift(-7) - hist['Close']) / hist['Close']
        hist['label'] = (hist['future_return'] > 0.05).astype(int)

        df = hist.drop("future_return", axis=1).dropna()
        y = df.label.values
        X = df.drop(columns="label").values

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.5)
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)

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
            max_epochs=20,
            hyperband_iterations=2,
            directory='untitled_project',
            project_name='tune_' + ticker_symbol
        )

        tuner.search(X_train_scaled, y_train, epochs=20, validation_data=(X_test_scaled, y_test), verbose=0)

        best_model = tuner.get_best_models(1)[0]
        best_model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2, verbose=0)

        y_pred_probs = best_model.predict(X_test_scaled)
        y_pred_classes = (y_pred_probs > np.percentile(y_pred_probs, 90)).astype(int)

        report = classification_report(y_test, y_pred_classes, digits=3)

        if y_pred_classes[-1][0]:
            statement = '📈 The model predicts that the stock price will rise more than 5% in the next 7 trading days. Might be a good time to buy!'
        else:
            statement = '📉 The model predicts that the stock price will not gain more than 5% in the next 7 trading days. Maybe hold off for now...'

        disclaimer = '(Disclaimer: This is not financial advice.)'

        # Ensure static folder exists
        if not os.path.exists('static'):
            os.makedirs('static')

        # Plot last 90 days of Close and SMA_20
        fig, ax = plt.subplots(figsize=(12, 6))
        last_90 = df[['Close', 'SMA_20']].tail(90)
        last_90.plot(ax=ax, title=f"{ticker_symbol.upper()} - Last 90 Days (Close & 20-Day SMA)", linewidth=1.5)
        ax.set_ylabel("Price ($)")
        ax.set_xlabel("Date")
        plt.tight_layout()

        chart_path = f"static/{ticker_symbol.upper()}_chart.png"
        fig.savefig(chart_path)
        plt.close(fig)

        return report, statement, disclaimer, chart_path

    except Exception as e:
        return f"Error while running model for {ticker_symbol.upper()}: {str(e)}", "", "", ""

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    statement = ""
    disclaimer = ""
    selected_ticker = ""
    chart_path = ""
    if request.method == "POST":
        ticker = request.form.get("ticker")
        selected_ticker = ticker.upper()
        result, statement, disclaimer, chart_path = run_model_for_ticker(ticker)
    return render_template("index.html", result=result, statement=statement, disclaimer=disclaimer, ticker=selected_ticker, chart_path=chart_path)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
