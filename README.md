# project4

Stock Price Prediction Web App:

This project is a Flask-based web application that predicts whether a stock's price will increase by more than 5% in the next 7 trading days. It uses historical stock data, technical indicators, and a neural network optimized with hyperparameter tuning to generate predictions and visualizations.

Features:

- Predicts stock price movements based on historical data and technical indicators.
- Uses a neural network with hyperparameter tuning via Keras Tuner.
- Generates classification reports for model evaluation.
- Displays visual stock charts with recent trends.
- Simple, clean, and responsive web interface built with HTML.

Project Structure:

    /Model/:

    app.py: Main Flask application code (runs the model and web server)
    templates/index.html: Frontend HTML for the web interface
    static/*.png - Generated stock charts saved as PNG images
    README.md - Project documentation (this file)
    Slide Deck Link: https://docs.google.com/presentation/d/1vPft-Uf-SsTidd2UTbKKvpAsJB2hnj-nzbdMRTUDoMM/edit?usp=sharing

How It Works:

User Input: Enter a stock ticker symbol in the web interface.
Data Fetching: Retrieves up to 10 years of daily stock price data from Yahoo Finance.
Feature Engineering:
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
20-day Simple Moving Average (SMA)
Calculates 7-day future return to classify buy signals.

Model Training:

Scales data with StandardScaler.
Splits into train/test sets.
Builds and tunes a neural network using Keras Tuner's Hyperband.
Prediction & Output:
Predicts whether the stock will rise >5% in 7 days.
Displays a classification report, verdict, disclaimer, and stock chart.
