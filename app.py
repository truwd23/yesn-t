import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# Load data from Excel
data = pd.read_csv("setelah_outlier_oke.csv")  # Ganti "nama_file.xlsx" dengan nama file Excel yang berisi data

# Features and target variable
X = data.drop(["Harga","Nama"], axis=1)
y = data["Harga"]

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction function
def predict_price(jenis, listrik, akses_24_jam, ac, kasur, k_mandi_dalam, kloset_duduk, penjaga_kos,
                  pengurus_kos, cctv, wifi, tempat_ibadah, bank, rumah_sakit, universitas):
    input_data = np.array([jenis, listrik, akses_24_jam, ac, kasur, k_mandi_dalam, kloset_duduk, penjaga_kos,
                           pengurus_kos, cctv, wifi, tempat_ibadah, bank, rumah_sakit, universitas]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)
    return predicted_price[0]


# Function to get informative text based on input
def get_info_text(input_value, info_dict):
    return info_dict.get(input_value, "No information available.")

# Information dictionary for jenis and universitas
jenis_info = {
    0: "Pilih Antara 1 - 3 ",
    1: "Putra",
    2: "Putri",
    3: "Campur"
}


# Streamlit app
st.title("Estimasi Harga Kos using Multiple Linear Regression")

jenis = st.number_input("Jenis", min_value=0, max_value=3, step=1)
st.write("Jenis:", get_info_text(jenis, jenis_info))
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
st.write("Pilih 1 apabila memiliki salah satu fasilitas yang ada di atas selain jenis kos.")

if st.button("Estimasi"):
    # Simpan harga dan nama kos sebelumnya
    existing_prices = data['Harga']
    existing_names = data['Nama']

    # Predict harga dengan fungsi predict_price
    predicted_price = predict_price(jenis, listrik, akses_24_jam, ac, kasur, k_mandi_dalam, kloset_duduk,
                                    penjaga_kos, pengurus_kos, cctv, wifi, tempat_ibadah, bank, rumah_sakit, universitas)

    # Mengubah nilai prediksi menjadi integer
    predicted_price = int(predicted_price)

    st.success(f"Predicted Price: {predicted_price}")

    # Menentukan rentang harga berdasarkan predicted price
    lower_limit = predicted_price - 200000
    upper_limit = predicted_price + 200000

    # Memfilter data sesuai dengan rentang harga
    filtered_data = data[(data['Harga'] >= lower_limit) & (data['Harga'] <= upper_limit)]

    # Menampilkan data yang sesuai dengan batas maksimal 10 data
    max_display_rows = 10
    filtered_data_display = filtered_data.head(max_display_rows)

    # Menampilkan tabel dengan scrollbar
    st.dataframe(filtered_data_display[['Nama', 'Harga']], height=200)

    # Menampilkan pesan jika terdapat lebih dari 10 data
    if len(filtered_data) > max_display_rows:
        st.info(f"Menampilkan 10 dari {len(filtered_data)} data. Gunakan scrollbar untuk melihat data lebih lanjut.")

    y_pred = model.predict(X_test)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("Evaluasi Model:")
    st.write(f"MAPE: {mape:.2f}%")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")
