import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Pengumpulan Data
file_path = 'data_penjualan.csv'
data = pd.read_csv(file_path)
print("Data Awal:")
print(data.head())

# Data Cleaning
print("\nMemeriksa missing values:")
print(data.isnull().sum())

data = data.dropna()
data['Tanggal'] = pd.to_datetime(data['Tanggal'])

print("\nJenis Kelamin Unik:")
print(data['Jenis Kelamin'].unique())

print("\nJenis Barang Unik:")
print(data['Jenis Barang'].unique())

# Data Transformation
data['Jenis Kelamin'] = data['Jenis Kelamin'].map({'Pria': 1, 'Wanita': 0})
data['Total Penjualan'] = data['Jumlah Barang'] * data['Harga Satuan']

# Exploratory Data Analysis (EDA)
print("\nDistribusi Total Penjualan:")
sns.histplot(data['Total Penjualan'], bins=30)
plt.title('Distribusi Total Penjualan')
plt.show()

print("\nPola Pembelian Berdasarkan Jenis Kelamin:")
sns.countplot(data['Jenis Kelamin'])
plt.title('Pola Pembelian Berdasarkan Jenis Kelamin')
plt.show()

print("\nKorelasi Data:")
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Korelasi Data')
plt.show()

# Modeling Data
X = data[['Jenis Kelamin', 'Jumlah Barang', 'Harga Satuan']]
y = data['Total Penjualan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Validasi dan Tuning Model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'\nMAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# Interpretasi dan Penyajian Hasil
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# Deploy dan Monitoring
model_filename = 'model_penjualan.pkl'
joblib.dump(model, model_filename)
print(f'\nModel disimpan sebagai {model_filename}')

# Maintenance dan Iterasi
def monitor_model_performance(data, model):
    X_new = data[['Jenis Kelamin', 'Jumlah Barang', 'Harga Satuan']]
    y_new = data['Total Penjualan']
    y_new_pred = model.predict(X_new)
    new_mae = mean_absolute_error(y_new, y_new_pred)
    print(f'New MAE: {new_mae}')
    
    threshold = 10000  # contoh threshold untuk MAE
    if new_mae > threshold:
        model.fit(X_new, y_new)
        joblib.dump(model, model_filename)
        print("Model retrained and saved.")

# Contoh pemanggilan fungsi monitoring
monitor_model_performance(data, model)
