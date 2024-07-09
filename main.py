import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

# Load stock data
@st.cache
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
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mae, mse, r2

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

day = st.number_input("Day", min_value=1, max_value=31, value=1)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
year = st.number_input("Year", min_value=2000, max_value=2023, value=2023)

if st.button("Predict"):
    predictions = predict_future_price(day, month, year)
    st.write(f"Predicted stock prices for {day}-{month}-{year}:")
    st.write(f"SVM Prediction: {predictions['SVM_Prediction']}")
    st.write(f"KNN Prediction: {predictions['KNN_Prediction']}")
    st.write(f"RF Prediction: {predictions['RF_Prediction']}")

# Evaluate models
svm_mae, svm_mse, svm_r2 = evaluate_model(svm_model, X_test_scaled, y_test)
knn_mae, knn_mse, knn_r2 = evaluate_model(knn_model, X_test_scaled, y_test)
rf_mae, rf_mse, rf_r2 = evaluate_model(rf_model, X_test_scaled, y_test)

st.write("### Model Evaluation")
st.write("#### SVM Model")
st.write(f"Mean Absolute Error: {svm_mae}")
st.write(f"Mean Squared Error: {svm_mse}")
st.write(f"R² Score: {svm_r2}")

st.write("#### KNN Model")
st.write(f"Mean Absolute Error: {knn_mae}")
st.write(f"Mean Squared Error: {knn_mse}")
st.write(f"R² Score: {knn_r2}")

st.write("#### Random Forest Model")
st.write(f"Mean Absolute Error: {rf_mae}")
st.write(f"Mean Squared Error: {rf_mse}")
st.write(f"R² Score: {rf_r2}")

# Correlation heatmap
st.write("### Correlation Heatmap")
st.write(sns.heatmap(df_stock.corr(), annot=True, cmap='coolwarm'))

# Show stock data
st.write("### Stock Data")
st.write(df_stock.head())
