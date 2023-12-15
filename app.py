import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Fungsi untuk memprediksi harga dengan Linear Regression


def predict_price(model, fitur_kos):
    harga_prediksi = model.predict(fitur_kos)[0]
    return harga_prediksi

# Fungsi untuk menghitung Mean Absolute Percentage Error (MAPE)


def calculate_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


# Memuat data dari file CSV
file_path = 'data.csv'
data_kos = pd.read_csv(file_path)
data_kos['K. Mandi Dalam'] = data_kos['Fasilitas'].apply(
    lambda x: 1 if 'K. Mandi Dalam' in x else 0)
data_kos['WiFi'] = data_kos['Fasilitas'].apply(
    lambda x: 1 if 'WiFi' in x else 0)
data_kos['Kasur'] = data_kos['Fasilitas'].apply(
    lambda x: 1 if 'Kasur' in x else 0)
data_kos['AC'] = data_kos['Fasilitas'].apply(lambda x: 1 if 'AC' in x else 0)
data_kos['Kloset Duduk'] = data_kos['Fasilitas'].apply(
    lambda x: 1 if 'Kloset Duduk' in x else 0)
data_kos['Akses 24 Jam'] = data_kos['Fasilitas'].apply(
    lambda x: 1 if 'Akses 24 Jam' in x else 0)

# Mengganti jenis kos menjadi campur = 1, peria = 2, dan wanita = 3
data_kos['Jenis'] = data_kos['Jenis'].map(
    {'Campur': 1, 'Putra': 2, 'Putri': 3})

# Menghitung prediksi harga
X = data_kos[['Jenis', 'K. Mandi Dalam', 'WiFi',
              'AC', 'Kasur', 'Kloset Duduk', 'Akses 24 Jam']]
y = np.log1p(data_kos['Harga'])  # Log-transformasi variabel target

# Standardisasi fitur-fitur numerik
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Menggunakan Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Tampilan Streamlit
st.title('Aplikasi Prediksi Harga Kos')
st.write('Gunakan formulir di bawah untuk memasukkan fitur kos dan memprediksi harga.')

# Formulir input untuk fitur kos
jenis_kos_mapping = {'Campur': 1, 'Putra': 2, 'Putri': 3}
jenis_kos = st.selectbox('Jenis Kos', list(jenis_kos_mapping.keys()))
k_mandi_dalam = st.selectbox('K. Mandi Dalam', ['Ya', 'Tidak'])
wifi = st.selectbox('WiFi', ['Ya', 'Tidak'])
ac = st.selectbox('AC', ['Ya', 'Tidak'])
kasur = st.selectbox('Kasur', ['Ya', 'Tidak'])
kloset_duduk = st.selectbox('Kloset Duduk', ['Ya', 'Tidak'])
akses_24_jam = st.selectbox('Akses 24 Jam', ['Ya', 'Tidak'])

# Mengubah nilai 'Ya' dan 'Tidak' menjadi 1 dan 0
jenis_kos_value = jenis_kos_mapping[jenis_kos]
fitur_kos = np.array([
    [jenis_kos_value, 1 if k_mandi_dalam == 'Ya' else 0, 0 if wifi == 'Ya' else 1,
     1 if ac == 'Ya' else 0, 0 if kasur == 'Ya' else 1, 1 if kloset_duduk == 'Ya' else 0,
     0 if akses_24_jam == 'Ya' else 1]
])

# Standardisasi fitur input
fitur_kos = scaler.transform(fitur_kos)

# Prediksi harga
harga_prediksi = np.expm1(predict_price(
    model, fitur_kos))  # Transformasi invers log

# Menampilkan hasil prediksi
st.subheader('Hasil Prediksi Harga')
st.write(f'Harga yang diprediksi: Rp {harga_prediksi:,.0f}')

# Menampilkan metrik evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = calculate_mape(np.expm1(y_test), np.expm1(y_pred)
                      )  # Transformasi invers log

st.subheader('Evaluasi Model')
st.write(f'Mean Squared Error (MSE): {mse:.2f}')
st.write(f'R-squared (R2): {r2:.2f}')
st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
