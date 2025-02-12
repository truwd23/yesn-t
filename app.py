import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Load data
data = pd.read_csv("setelah_outlier_oke.csv")

# Features and target variable
X = data.drop(["Harga", "Nama"], axis=1)
y = data["Harga"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Model Evaluation
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

def predict_price(features):
    input_data = np.array(features).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)[0]

# Streamlit App
st.title("Estimasi Harga Kos using Multiple Linear Regression")

jenis = st.number_input("Jenis", min_value=0, max_value=3, step=1)
listrik = st.number_input("Listrik", min_value=0, max_value=1, step=1)
akses_24_jam = st.number_input("Akses 24 Jam", min_value=0, max_value=1, step=1)
ac = st.number_input("AC", min_value=0, max_value=1, step=1)
kasur = st.number_input("Kasur", min_value=0, max_value=1, step=1)
k_mandi_dalam = st.number_input("K. Mandi Dalam", min_value=0, max_value=1, step=1)
kloset_duduk = st.number_input("Kloset Duduk", min_value=0, max_value=1, step=1)
penjaga_kos = st.number_input("Penjaga Kos", min_value=0, max_value=1, step=1)
pengurus_kos = st.number_input("Pengurus Kos", min_value=0, max_value=1, step=1)
cctv = st.number_input("CCTV", min_value=0, max_value=1, step=1)
wifi = st.number_input("WiFi", min_value=0, max_value=1, step=1)
tempat_ibadah = st.number_input("Tempat Ibadah", min_value=0, max_value=1, step=1)
bank = st.number_input("Bank", min_value=0, max_value=1, step=1)
rumah_sakit = st.number_input("Rumah Sakit", min_value=0, max_value=1, step=1)
universitas = st.number_input("Universitas", min_value=0, max_value=1, step=1)

if st.button("Estimasi"):
    predicted_price = predict_price([jenis, listrik, akses_24_jam, ac, kasur, k_mandi_dalam, kloset_duduk,
                                     penjaga_kos, pengurus_kos, cctv, wifi, tempat_ibadah, bank, rumah_sakit, universitas])
    predicted_price = int(predicted_price)
    st.success(f"Predicted Price: {predicted_price}")
    
    lower_limit = predicted_price - 200000
    upper_limit = predicted_price + 200000
    filtered_data = data[(data['Harga'] >= lower_limit) & (data['Harga'] <= upper_limit)]
    st.dataframe(filtered_data[['Nama', 'Harga']], height=200, width=800)
    
    st.subheader("Evaluasi Model:")
    st.write(f"R^2 Score (Training): {r2_train:.2f}")
    st.write(f"R^2 Score (Testing): {r2_test:.2f}")
    st.write(f"RMSE (Training): {rmse_train:.2f}")
    st.write(f"RMSE (Testing): {rmse_test:.2f}")
    st.write(f"MAPE (Training): {mape_train:.2%}")
    st.write(f"MAPE (Testing): {mape_test:.2%}")
