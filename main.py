import subprocess
import sys

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

install_packages()

import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from datetime import datetime

# Load stock data
@st.cache_data
def load_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    df_stock = stock.history(period=period)
    df_stock.reset_index(inplace=True)
    df_stock['Day'] = df_stock['Date'].dt.day
    df_stock['Month'] = df_stock['Date'].dt.month
    df_stock['Year'] = df_stock['Date'].dt.year
    return df_stock

# Load data for NVIDIA
df_stock = load_stock_data("NVDA")

features = ['Day', 'Month', 'Year']
target = 'Close'

X = df_stock[features]
y = df_stock[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVR()
svm_model.fit(X_train_scaled, y_train)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    return r2

def predict_future_price(day, month, year):
    input_features = pd.DataFrame([[day, month, year]], columns=features)
    input_features_scaled = scaler.transform(input_features)

    svm_prediction = svm_model.predict(input_features_scaled)[0]
    knn_prediction = knn_model.predict(input_features_scaled)[0]
    rf_prediction = rf_model.predict(input_features_scaled)[0]

    return {
        'SVM_Prediction': svm_prediction,
        'KNN_Prediction': knn_prediction,
        'RF_Prediction': rf_prediction,
    }

st.title("NVIDIA Stock Price Prediction")

svm_r2 = evaluate_model(svm_model, X_test_scaled, y_test)
knn_r2 = evaluate_model(knn_model, X_test_scaled, y_test)
rf_r2 = evaluate_model(rf_model, X_test_scaled, y_test)

model_evaluation = {
    "Model": ["SVM", "KNN", "Random Forest"],
    "Accuracy": [f"{round(svm_r2, 1)}%", f"{round(knn_r2, 1)}%", f"{round(rf_r2, 1)}%"]
}

df_evaluation = pd.DataFrame(model_evaluation)

st.write("### Model Evaluation")
st.table(df_evaluation)

# Date input
date_input = st.date_input("Select Date", min_value=datetime(1999, 1, 22), value=datetime.now())

day = date_input.day
month = date_input.month
year = date_input.year

if st.button("Predict"):
    predictions = predict_future_price(day, month, year)
    st.write(f"Predicted stock prices for {day}-{month}-{year}:")
    st.write(f"SVM Prediction: {predictions['SVM_Prediction']:.2f}")
    st.write(f"KNN Prediction: {predictions['KNN_Prediction']:.2f}")
    st.write(f"RF Prediction: {predictions['RF_Prediction']:.2f}")
