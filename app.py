import pandas as pd
import numpy as np
import math
import gradio as gr
import plotly.graph_objects as go
import tempfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from docx import Document

model = None


# ------------------- DOCX -------------------
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


# ------------------- CLEAN -------------------
def clean_dataframe(df):
    df = df.loc[:, ~df.columns.duplicated()]

    rename_map = {}
    if 'Дата' in df.columns:
        rename_map['Дата'] = 'date'
    if 'Потребление' in df.columns:
        rename_map['Потребление'] = 'consumption'

    df = df.rename(columns=rename_map)
    return df


# ------------------- SEQUENCES -------------------
def create_sequences(data, seq_length=12):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)


# ------------------- MODEL -------------------
def train_and_predict(df):
    global model

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    df = df.dropna(subset=['date', 'consumption'])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['consumption']])

    X, y = create_sequences(scaled, 12)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model is None:
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(12, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)

    y_pred = model.predict(X_test, verbose=0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    last_seq = scaled[-12:].reshape(1, 12, 1)

    future = []
    for _ in range(12):
        p = model.predict(last_seq, verbose=0)
        future.append(p[0, 0])
        last_seq = np.append(last_seq[:, 1:, :], [[[p[0,0]]]], axis=1)

    future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

    dates = pd.date_range(df['date'].max(), periods=12, freq='MS').strftime('%Y-%m')

    return dates, future, mae, rmse


# ------------------- ANOMALIES -------------------
def detect_anomalies(preds, threshold=0.2):
    anomalies = [False]
    for i in range(1, len(preds)):
        change = abs(preds[i] - preds[i-1]) / preds[i-1]
        anomalies.append(change > threshold)
    return anomalies


# ------------------- EXPORT -------------------
def export_excel(dates, preds):
    df = pd.DataFrame({
        "Month": dates,
        "Prediction": preds
    })

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(tmp.name, index=False)

    return tmp.name

def export(dates, preds):
    if dates is None or preds is None:
        return None
    return export_excel(dates, preds)


# ------------------- MAIN FUNCTION -------------------
def run_model(file):
    if file is None:
        return None, "Нет файла", None, None, None

    if file.name.endswith(".csv"):
        df = pd.read_csv(file.name)
    else:
        df = docx_to_dataframe(file)

    df = clean_dataframe(df)

    dates, preds, mae, rmse = train_and_predict(df)
    anomalies = detect_anomalies(preds)

    result_df = pd.DataFrame({
        "date": dates,
        "prediction": preds,
        "anomaly": anomalies
    })

    # ------------------- PLOTLY GRAPH -------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=preds,
        mode='lines+markers',
        name='Forecast'
    ))

    fig.add_trace(go.Scatter(
        x=[dates[i] for i in range(len(dates)) if anomalies[i]],
        y=[preds[i] for i in range(len(dates)) if anomalies[i]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Anomaly'
    ))

    fig.update_layout(
        title="Energy Consumption Forecast",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Consumption"
    )

    return result_df, f"MAE: {mae:.4f}, RMSE: {rmse:.4f}", fig, dates, preds


# ------------------- UI (BLOCKS) -------------------
with gr.Blocks() as demo:

    gr.Markdown("# ⚡ Energy AI Forecast")

    file_input = gr.File(label="Upload CSV or DOCX")

    run_btn = gr.Button("Run 🚀")
    export_btn = gr.Button("Export Excel 📥")

    table = gr.Dataframe()
    metrics = gr.Textbox()
    chart = gr.Plot()

    # ✅ СНАЧАЛА state!
    state_dates = gr.State()
    state_preds = gr.State()

    # ✅ download компонент
    download_file = gr.File(label="Download Excel")

    # ---------------- RUN ----------------
    def run(file):
        return run_model(file)

    run_btn.click(
        run,
        inputs=file_input,
        outputs=[table, metrics, chart, state_dates, state_preds]
    )

    # ---------------- EXPORT ----------------
    def export(dates, preds):
        if dates is None or preds is None:
            return None
        return export_excel(dates, preds)

    export_btn.click(
        export,
        inputs=[state_dates, state_preds],
        outputs=download_file
    )

demo.launch(server_name="0.0.0.0", server_port=7860)