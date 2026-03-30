from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from docx import Document
import math
import os
app = Flask(__name__)


# ------------------- Обработка Word -------------------
def docx_to_dataframe(file):
    doc = Document(file)
    table = doc.tables[0]
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    data = []
    for row in table.rows[1:]:
        data.append([cell.text.strip() for cell in row.cells])
    df = pd.DataFrame(data, columns=headers)
    # Преобразуем числовые значения
    if 'Потребление' in df.columns:
        df['Потребление'] = df['Потребление'].str.replace(',', '.').astype(float)
        df = df.rename(columns={"Потребление": "consumption", "Дата": "date"})
    elif 'consumption' in df.columns:
        df['consumption'] = df['consumption'].astype(float)
    return df


# ------------------- Чистка DataFrame -------------------
def clean_dataframe(df):
    df = df.loc[:, ~df.columns.duplicated()]  # убираем дубли
    rename_map = {}
    if 'Дата' in df.columns:
        rename_map['Дата'] = 'date'
    if 'Потребление' in df.columns:
        rename_map['Потребление'] = 'consumption'
    df = df.rename(columns=rename_map)
    return df


# ------------------- Создание последовательностей -------------------
def create_sequences(data, seq_length=12):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)


# ------------------- Тренировка LSTM и прогноз -------------------
def train_and_predict(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    df = df.dropna(subset=['date', 'consumption'])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['consumption']])

    X, y = create_sequences(scaled, seq_length=12)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Разделение на train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)

    # Предсказания на тесте
    y_pred = model.predict(X_test, verbose=0)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    # Прогноз на 12 месяцев вперед
    last_seq = scaled[-12:].reshape((1, 12, 1))
    preds_scaled = []
    for _ in range(12):
        pred = model.predict(last_seq, verbose=0)
        pred = pred.reshape((1, 1, 1))
        preds_scaled.append(float(pred[0, 0, 0]))
        last_seq = np.append(last_seq[:, 1:, :], pred, axis=1)

    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

    # Преобразуем прогноз обратно в исходный масштаб
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    return [d.strftime('%Y-%m') for d in future_dates], preds.tolist(), mae, rmse


# ------------------- Выявление аномалий -------------------
def detect_anomalies(preds, threshold=0.2):
    anomalies = [False]
    for i in range(1, len(preds)):
        change = abs(preds[i] - preds[i - 1]) / preds[i - 1]
        anomalies.append(change > threshold)
    anomalies = [bool(a) for a in anomalies]
    return anomalies


# ------------------- Маршруты Flask -------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "Нет файла"}), 400
    file = request.files['file']

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.docx'):
            df = docx_to_dataframe(file)
        else:
            return jsonify({"error": "Неподдерживаемый формат файла"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    df = clean_dataframe(df)

    if 'date' not in df.columns or 'consumption' not in df.columns:
        return jsonify({"error": "Файл должен содержать колонки date и con-sumption"}), 400

    dates, preds, mae, rmse = train_and_predict(df)
    anomalies = detect_anomalies(preds)

    return jsonify({
        "dates": dates,
        "predictions": preds,
        "anomalies": anomalies,
        "mae": mae,
        "rmse": rmse
    })


# ------------------- Экспорт Excel -------------------
@app.route("/export_excel", methods=["POST"])
def export_excel():
    data = request.get_json()
    df = pd.DataFrame({
        "Месяц": data["dates"],
        "Прогноз": data["predictions"]
    })
    file_path = "forecast.xlsx"
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)


# ------------------- Запуск сервера -------------------


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

