import pandas as pd
import numpy as np
import math
import os
import gradio as gr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from docx import Document
model = None
scaler = None

# ------------------- Обработка Word -------------------
def docx_to_dataframe(file):
    doc = Document(file.name)
    table = doc.tables[0]

    headers = [cell.text.strip() for cell in table.rows[0].cells]

    data = []
    for row in table.rows[1:]:
        data.append([cell.text.strip() for cell in row.cells])

    df = pd.DataFrame(data, columns=headers)

    if 'Потребление' in df.columns:
        df['Потребление'] = df['Потребление'].str.replace(',', '.').astype(float)
        df = df.rename(columns={"Потребление": "consumption", "Дата": "date"})
    elif 'consumption' in df.columns:
        df['consumption'] = df['consumption'].astype(float)

    return df
# ------------------- Чистка DataFrame -------------------
def clean_dataframe(df):
    df = df.loc[:, ~df.columns.duplicated()]

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
    global model, scaler

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    df = df.dropna(subset=['date', 'consumption'])

    # scaler ONLY ONCE per file (но не глобально пересоздаём модель)
    scaler_local = MinMaxScaler()
    scaled = scaler_local.fit_transform(df[['consumption']])

    X, y = create_sequences(scaled, seq_length=12)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 🔥 MODEL: train ONLY ONCE per HF runtime
    if model is None:
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)

    y_pred = model.predict(X_test, verbose=0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    last_seq = scaled[-12:].reshape((1, 12, 1))

    preds_scaled = []
    for _ in range(12):
        pred = model.predict(last_seq, verbose=0)
        pred = pred.reshape((1, 1, 1))
        preds_scaled.append(float(pred[0, 0, 0]))
        last_seq = np.append(last_seq[:, 1:, :], pred, axis=1)

    last_date = df['date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )

    preds = scaler_local.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()

    return (
        [d.strftime('%Y-%m') for d in future_dates],
        preds.tolist(),
        mae,
        rmse
    )
# ------------------- Аномалии -------------------
def detect_anomalies(preds, threshold=0.2):
    anomalies = [False]

    for i in range(1, len(preds)):
        change = abs(preds[i] - preds[i - 1]) / preds[i - 1]
        anomalies.append(change > threshold)

    return [bool(a) for a in anomalies]

# ------------------- GRADIO LOGIC -------------------
def run_model(file):
    if file is None:
        return "Нет файла"

    if file.name.endswith(".csv"):
        df = pd.read_csv(file.name)
    elif file.name.endswith(".docx"):
        df = docx_to_dataframe(file)
    else:
        return "Только CSV или DOCX"

    df = clean_dataframe(df)

    if 'date' not in df.columns or 'consumption' not in df.columns:
        return "Нужны колонки: date и consumption"

    dates, preds, mae, rmse = train_and_predict(df)
    anomalies = detect_anomalies(preds)

    result_df = pd.DataFrame({
        "date": dates,
        "prediction": preds,
        "anomaly": anomalies
    })

    return result_df, f"MAE: {mae:.4f}, RMSE: {rmse:.4f}"
# ------------------- UI -------------------
demo = gr.Interface(
    fn=run_model,
    inputs=gr.File(),
    outputs=[gr.Dataframe(), gr.Textbox()],
    title="Energy AI Forecast ⚡"
)
demo.launch(server_name="0.0.0.0", server_port=7860)