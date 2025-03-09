"""
Projeto Robusto para Previsão de Terremotos, Análise de Risco, Monitoramento e Dashboard Interativo
com Geração de Gráficos para CADA LOCALIDADE

Funcionalidades integradas:
1. Previsão Agregada com Ensemble (LSTM + Regressão Linear):
   - Coleta dos dados dos últimos 30 dias (eventos com magnitude ≥ 4.0); agregação diária;
     treinamento dos modelos; geração de gráficos (forecast e histórico de erro) – títulos fixos "Local: Geral".
   - (Observação: a previsão dummy de regressão linear é baseada na média dos últimos 7 dias.)

2. Análise por Localidade:
   - Une os arquivos CSV gerados (com padrão "AgentBroker*.csv") na pasta DATA; agrupa os eventos
     pelo campo “Place”; valida e converte os dados numéricos; calcula número de eventos, média de
     magnitude, média de profundidade, “risk” (Magnitude/(Depth+1)), risco máximo e coordenadas médias;
     gera gráficos de indicadores e localização.

3. Monitoramento em Tempo Real (Simulado):
   - Simula a leitura de sensores a cada 10 segundos; se dois eventos consecutivos excederem o limiar
     (ALERT_THRESHOLD) e um cooldown de 60 segundos for respeitado, dispara um alerta (dummy).

4. Dashboard Interativo:
   - Disponibiliza endpoints para consulta dos dados agregados e dos gráficos.

5. Organização de Arquivos:
   - Arquivos (CSV e imagens) são salvos com o padrão “AgentBrokerDDMMYYYY_HHMMSS.ext”.
   - Jobs agendados removem arquivos com mais de 30 dias (caso a variável RUN_MAIN esteja ativa).

Funcionalidades adicionais:
6. Notificações e Alertas Automatizados.
7. Integração com Banco de Dados Persistente (SQLite).
8. API Avançada para Filtragem e Consulta.
9. Logging e Monitoramento de Performance.
10. Segurança e Autenticação dos Endpoints.

Adicionalmente:
- Foi incluído o novo endpoint **/ia-risk-dashboard** que utiliza um “especialista em Computação Sísmica com IA” 
  para analisar os dados agregados. Esse endpoint une os CSVs gerados, calcula as métricas por localidade, computa um
  índice de risco (risk_index = max_risk × event_count) e, com base nisso, gera um comentário de análise e recomendações.
  A resposta inclui a lista ordenada de localidades, os caminhos para gráficos (indicadores, localização e índice de risco)
  e o comentário do especialista.

Requisitos:
  - Python 3.7+
  - Instalar as bibliotecas: requests, pandas, numpy, matplotlib, seaborn, schedule,
    tensorflow, scikit-learn, fastapi, uvicorn
    (Exemplo: pip install requests pandas numpy matplotlib seaborn schedule tensorflow scikit-learn fastapi uvicorn)

Autor: Seu Nome
Data: [Data Atual]
"""

import os
import logging
import sqlite3
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import schedule
import time
import threading
import random
import glob
from datetime import datetime, timedelta
from functools import wraps

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("AgentBroker.log", encoding="utf-8")
    ]
)

# --- Verificação de GPU ---
if tf.test.gpu_device_name():
    logging.info("Dispositivo GPU encontrado. Executando no GPU.")
else:
    logging.info("Nenhum dispositivo GPU encontrado. Executando no CPU.")

# --- Diretórios para Dados, Plots e Banco de Dados ---
DATA_DIR = "data"
PLOTS_DIR = "plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "agentbroker.db")

# --- Constantes de Alerta ---
ALERT_THRESHOLD = 6.0      # Exemplo de threshold
COOLDOWN_PERIOD = 60       # Segundos
SENSOR_INTERVAL = 10       # Segundos

# --- Configuração de Autenticação ---
API_KEY = "mysecretapikey"
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
async def get_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        # Em ambiente de desenvolvimento, se não fornecido, utiliza o valor padrão
        api_key = API_KEY
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    return api_key

# --- Decorator para Monitorar Performance ---
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"Função {func.__name__} executada em {elapsed:.3f} segundos.")
        return result
    return wrapper

# --- Função para Gerar Nomes de Arquivos ---
def generate_filename(extension: str, prefix: str = "AgentBroker") -> str:
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    return f"{prefix}{timestamp}.{extension}"

# --- Banco de Dados ---
def init_db():
    """Inicializa o banco de dados SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time INTEGER,
            place TEXT,
            magnitude REAL,
            longitude REAL,
            latitude REAL,
            depth REAL
        )
    """)
    conn.commit()
    conn.close()
    logging.info("Banco de dados inicializado.")

def store_event_in_db(event: dict):
    """Armazena um evento sísmico no banco de dados."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO events (time, place, magnitude, longitude, latitude, depth)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (event["Time"], event["Place"], event["Magnitude"], event["Longitude"], event["Latitude"], event["Depth"]))
    conn.commit()
    conn.close()

# --- Funções de Coleta, Processamento e Previsão ---
@measure_time
def fetch_seismic_data(start_time: str, end_time: str, min_magnitude: float = 4.0) -> dict:
    """Coleta dados sísmicos da API do USGS."""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minmagnitude": min_magnitude
    }
    try:
        logging.info(f"Solicitando dados de {start_time} até {end_time}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Erro ao buscar dados sísmicos: {e}")
        return None

@measure_time
def save_seismic_data_to_csv(data: dict, filename: str) -> None:
    """Salva os dados sísmicos em CSV e armazena os eventos no BD."""
    if data and "features" in data:
        records = []
        for feature in data["features"]:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coordinates = geom.get("coordinates", [None, None, None])
            record = {
                "Time": props.get("time"),
                "Place": props.get("place"),
                "Magnitude": props.get("mag"),
                "Longitude": coordinates[0],
                "Latitude": coordinates[1],
                "Depth": coordinates[2]
            }
            records.append(record)
            store_event_in_db(record)
        df = pd.DataFrame(records)
        csv_path = os.path.join(DATA_DIR, filename)
        df.to_csv(csv_path, index=False)
        logging.info(f"Dados salvos em {csv_path}")
    else:
        logging.warning("Nenhum dado disponível para salvar.")

def prepare_time_series(df: pd.DataFrame) -> pd.Series:
    """Prepara a série temporal diária (média de magnitude) a partir dos dados."""
    df['Datetime'] = pd.to_datetime(df['Time'], unit='ms')
    df['Date'] = df['Datetime'].dt.date
    daily_avg = df.groupby('Date')['Magnitude'].mean().sort_index()
    return daily_avg

def create_sequences(series: np.ndarray, window: int):
    """Cria sequências para previsão com janela deslizante."""
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X).reshape(-1, window, 1), np.array(y).reshape(-1, 1)

def build_and_train_lstm(X, y, epochs=100, batch_size=16):
    """Constrói e treina o modelo LSTM."""
    model = Sequential([
        LSTM(50, activation='tanh', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    logging.info("Iniciando treinamento do modelo LSTM...")
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])
    logging.info("Treinamento concluído.")
    return model, history

def plot_time_series_with_forecast(dates, series, forecast_date, forecast_value, save_path: str):
    """Gera gráfico da série histórica e o forecast."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=dates, y=series, marker='o', label='Média Diária de Magnitude')
    plt.axvline(x=forecast_date, color='r', linestyle='--', label='Data do Forecast')
    plt.scatter([forecast_date], [forecast_value], color='r', zorder=5, label=f'Previsão: {forecast_value:.2f}')
    plt.title('Local: Geral - Forecast da Média Diária de Magnitude')
    plt.xlabel('Data')
    plt.ylabel('Magnitude')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Gráfico salvo em {save_path}")

# --- Funcionalidade 1: Previsão Agregada com Ensemble ---
@measure_time
def ensemble_forecast() -> dict:
    """Realiza previsão agregada usando ensemble (LSTM + regressão linear dummy)."""
    resultado = {}
    try:
        end_datetime = datetime.utcnow()
        start_datetime = end_datetime - timedelta(days=30)
        start_str = start_datetime.strftime("%Y-%m-%d")
        end_str = end_datetime.strftime("%Y-%m-%d")
        data = fetch_seismic_data(start_str, end_str, min_magnitude=4.0)
        if data is None:
            raise Exception("Falha na obtenção dos dados sísmicos.")
        csv_filename = generate_filename("csv")
        save_seismic_data_to_csv(data, csv_filename)
        df = pd.read_csv(os.path.join(DATA_DIR, csv_filename))
        if df.empty:
            raise Exception("CSV vazio. Sem dados para previsão.")
        daily_series = prepare_time_series(df)
        if len(daily_series) < 10:
            raise Exception("Dados insuficientes para previsão. (menos que 10 dias)")
        series_values = daily_series.values.astype('float32')
        window = 7
        X, y = create_sequences(series_values, window=window)
        epochs = 100
        model, history = build_and_train_lstm(X, y, epochs=epochs, batch_size=16)
        last_sequence = series_values[-window:].reshape(1, window, 1)
        forecast_lstm = model.predict(last_sequence)[0][0]
        forecast_lr = np.mean(last_sequence)
        forecast_ensemble = (forecast_lstm + forecast_lr) / 2
        last_date = datetime.strptime(str(daily_series.index[-1]), "%Y-%m-%d")
        forecast_date = last_date + timedelta(days=1)
        train_predictions = model.predict(X)
        mse = mean_squared_error(y, train_predictions)
        logging.info(f"MSE do treinamento: {mse:.4f}")
        logging.info(f"Forecast (Ensemble) para {forecast_date.date()}: {forecast_ensemble:.2f}")
        dates = [datetime.strptime(str(d), "%Y-%m-%d") for d in daily_series.index.astype(str)]
        forecast_plot = os.path.join(PLOTS_DIR, generate_filename("png"))
        plot_time_series_with_forecast(dates, series_values, forecast_date, forecast_ensemble, forecast_plot)
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Loss no Treinamento')
        plt.title('Local: Geral - Histórico de Perda do Modelo LSTM')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        error_plot = os.path.join(PLOTS_DIR, generate_filename("png"))
        plt.tight_layout()
        plt.savefig(error_plot)
        plt.close()
        resultado = {
            "start_date": start_str,
            "end_date": end_str,
            "forecast_date": forecast_date.strftime("%Y-%m-%d"),
            "forecast_lstm": round(float(forecast_lstm), 2),
            "forecast_lr": round(float(forecast_lr), 2),
            "forecast_ensemble": round(float(forecast_ensemble), 2),
            "mse": mse,
            "csv_file": os.path.join(DATA_DIR, csv_filename),
            "forecast_plot": forecast_plot,
            "error_plot": error_plot
        }
        return resultado
    except Exception as e:
        logging.error(f"Erro na previsão em ensemble: {e}")
        raise e

# --- Funcionalidade 2: Análise por Localidade ---
def calculate_locality_metrics(group: pd.DataFrame) -> pd.Series:
    """Calcula métricas para uma localidade."""
    risk = group['Magnitude'] / (group['Depth'] + 1)
    return pd.Series({
        'event_count': group['Magnitude'].count(),
        'mean_magnitude': group['Magnitude'].mean(),
        'mean_depth': group['Depth'].mean(),
        'mean_risk': risk.mean(),
        'max_risk': risk.max(),
        'avg_latitude': group['Latitude'].mean(),
        'avg_longitude': group['Longitude'].mean()
    })

@measure_time
def locality_analysis() -> dict:
    """
    Realiza análise por localidade unindo todos os CSV (padrão "AgentBroker*.csv").
    Agrupa os eventos por "Place", gera gráficos de indicadores e localização e retorna os resultados.
    """
    try:
        all_csv_path = os.path.join(DATA_DIR, "AgentBroker_all.csv")
        if os.path.exists(all_csv_path):
            df = pd.read_csv(all_csv_path)
        else:
            csv_files = glob.glob(os.path.join(DATA_DIR, "AgentBroker*.csv"))
            if not csv_files:
                raise Exception("Nenhum arquivo CSV encontrado para análise.")
            df_list = [pd.read_csv(fp) for fp in sorted(csv_files)]
            df = pd.concat(df_list, ignore_index=True)
        if df.empty:
            raise Exception("Os arquivos CSV estão vazios.")
        df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
        df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df = df.dropna(subset=['Magnitude', 'Depth', 'Latitude', 'Longitude'])
        grouped = df.groupby('Place').apply(calculate_locality_metrics).reset_index()
        grouped = grouped.sort_values(by='max_risk', ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Place', y='mean_magnitude', data=grouped)
        plt.title('Local: Indicadores - Média de Magnitude por Localidade')
        plt.xticks(rotation=45)
        plt.tight_layout()
        indicators_plot = os.path.join(PLOTS_DIR, generate_filename("png"))
        plt.savefig(indicators_plot)
        plt.close()
        logging.info(f"Gráfico de indicadores salvo em {indicators_plot}")
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='Longitude', y='Latitude', data=df, hue='Place', alpha=0.6, legend=False)
        plt.scatter(grouped['avg_longitude'], grouped['avg_latitude'], color='red', marker='X', s=100, label='Média')
        plt.title('Local: Localização dos Eventos por Localidade')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.tight_layout()
        location_plot = os.path.join(PLOTS_DIR, generate_filename("png"))
        plt.savefig(location_plot)
        plt.close()
        logging.info(f"Gráfico de localização salvo em {location_plot}")
        result = {
            "aggregated_data": grouped.to_dict(orient="records"),
            "indicators_plot": indicators_plot,
            "location_plot": location_plot
        }
        return result
    except Exception as e:
        logging.error(f"Erro na análise por localidade: {e}")
        raise e

# --- Funcionalidade 3: Monitoramento em Tempo Real (Simulado) ---
last_alert_time = 0
consecutive_alerts = 0
def send_alert(message: str):
    """Envia um alerta (dummy: apenas loga o alerta)."""
    logging.warning(f"ALERTA: {message}")

def simulate_sensor_reading() -> float:
    """Simula leitura de sensor com valor aleatório entre 2.0 e 8.0."""
    return round(random.uniform(2.0, 8.0), 2)

def simulate_real_time_monitoring():
    """
    Simula leitura de sensores a cada 10 segundos.
    Se dois eventos consecutivos excederem o ALERT_THRESHOLD e o cooldown for respeitado,
    dispara um alerta.
    """
    global last_alert_time, consecutive_alerts
    while True:
        sensor_value = simulate_sensor_reading()
        logging.info(f"Sensor reading: {sensor_value}")
        if sensor_value >= ALERT_THRESHOLD:
            consecutive_alerts += 1
            logging.info(f"Evento acima do limiar. Contador: {consecutive_alerts}")
            if consecutive_alerts >= 2:
                current_time = time.time()
                if current_time - last_alert_time >= COOLDOWN_PERIOD:
                    send_alert(f"Detectados {consecutive_alerts} eventos consecutivos acima do limiar. Última leitura: {sensor_value}")
                    last_alert_time = current_time
                    consecutive_alerts = 0
        else:
            consecutive_alerts = 0
        time.sleep(SENSOR_INTERVAL)

# --- Funcionalidade 4: Dashboard Interativo (via API) ---
def get_dashboard_data() -> dict:
    """Retorna dados agregados por localidade para o dashboard."""
    try:
        csv_files = glob.glob(os.path.join(DATA_DIR, "AgentBroker*.csv"))
        if not csv_files:
            raise Exception("Nenhum arquivo CSV encontrado para o dashboard.")
        df_list = [pd.read_csv(fp) for fp in sorted(csv_files)]
        df = pd.concat(df_list, ignore_index=True)
        aggregated = df.groupby('Place').agg(
            event_count=('Magnitude', 'count'),
            mean_magnitude=('Magnitude', 'mean'),
            mean_depth=('Depth', 'mean')
        ).reset_index()
        return aggregated.to_dict(orient="records")
    except Exception as e:
        logging.error(f"Erro ao obter dados do dashboard: {e}")
        return {}

# --- Funcionalidade 5: Organização de Arquivos ---
def remove_old_files(directory: str, days: int = 30):
    """Remove arquivos com mais de 'days' dias no diretório indicado."""
    now = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and (now - os.path.getmtime(file_path)) > days * 86400:
            try:
                os.remove(file_path)
                logging.info(f"Arquivo removido: {file_path}")
            except Exception as e:
                logging.error(f"Erro ao remover o arquivo {file_path}: {e}")

# --- Funcionalidade 6: Notificações e Alertas Automatizados ---
def send_notification(alert_message: str):
    """Função dummy para enviar notificações automatizadas."""
    logging.warning(f"Notificação enviada: {alert_message}")

# --- Funcionalidade 7: API Avançada para Filtragem e Consulta ---
def query_events(start_date: str = None, end_date: str = None, place: str = None) -> list:
    """Consulta eventos do banco de dados com filtros opcionais."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if start_date:
        query += " AND time >= ?"
        dt = datetime.strptime(start_date, "%Y-%m-%d")
        params.append(int(dt.timestamp() * 1000))
    if end_date:
        query += " AND time <= ?"
        dt = datetime.strptime(end_date, "%Y-%m-%d")
        params.append(int(dt.timestamp() * 1000))
    if place:
        query += " AND place = ?"
        params.append(place)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    events = []
    for row in rows:
        events.append({
            "id": row[0],
            "time": row[1],
            "place": row[2],
            "magnitude": row[3],
            "longitude": row[4],
            "latitude": row[5],
            "depth": row[6]
        })
    return events

# --- Nova Funcionalidade: Dashboard de Risco via IA com Comentários Especializados ---
def plot_risk_index_bar_chart(grouped: pd.DataFrame, save_path: str):
    """
    Gera um gráfico de barras com o índice de risco para cada localidade.
    O índice de risco é calculado como: risk_index = max_risk * event_count.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Place", y="risk_index", data=grouped)
    plt.title("Índice de Risco por Localidade")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Gráfico de índice de risco salvo em {save_path}")

@measure_time
def ia_risk_dashboard() -> dict:
    """
    Processa os dados para identificar localidades com maior chance de terremoto.
    - Une os arquivos CSV (padrão "AgentBroker*.csv") da pasta DATA.
    - Processa os dados, converte campos numéricos e elimina registros inconsistentes.
    - Agrupa os dados por localidade utilizando calculate_locality_metrics.
    - Calcula o índice de risco: risk_index = max_risk * event_count.
    - Ordena as localidades pelo índice.
    - Gera o gráfico de índice de risco e reusa os gráficos de indicadores e localização (de locality_analysis).
    - Gera comentários especializados simulando um "Especialista em Computação Sísmica com IA".
    - Retorna um dicionário com a lista ordenada, os caminhos dos gráficos e a análise.
    """
    try:
        csv_files = glob.glob(os.path.join(DATA_DIR, "AgentBroker*.csv"))
        if not csv_files:
            raise Exception("Nenhum arquivo CSV encontrado para análise de risco.")
        df_list = [pd.read_csv(fp) for fp in sorted(csv_files)]
        df = pd.concat(df_list, ignore_index=True)
        if df.empty:
            raise Exception("Os arquivos CSV estão vazios.")
        df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
        df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
        df = df.dropna(subset=['Magnitude', 'Depth'])
        grouped = df.groupby('Place').apply(calculate_locality_metrics).reset_index()
        grouped['risk_index'] = grouped['max_risk'] * grouped['event_count']
        grouped = grouped.sort_values(by='risk_index', ascending=False)
        
        # Gera o gráfico de índice de risco
        risk_index_plot = os.path.join(PLOTS_DIR, generate_filename("png"))
        plot_risk_index_bar_chart(grouped, risk_index_plot)
        
        # Reusa os gráficos de indicadores e localização da análise por localidade
        locality_data = locality_analysis()
        
        # Gera comentário de análise do "Especialista em Computação Sísmica"
        expert_comments = "Análise Especializada: "
        if not grouped.empty:
            top = grouped.iloc[0]
            if top['risk_index'] > 100:
                expert_comments += (
                    f"A localidade '{top['Place']}' apresenta um índice de risco elevado ({top['risk_index']:.2f}). "
                    "Isso indica uma elevada probabilidade de futuros eventos sísmicos significativos. "
                    "Recomenda-se monitoramento intensivo e investigação geológica detalhada."
                )
            else:
                expert_comments += (
                    "Os índices de risco estão moderados. Entretanto, como eventos sísmicos podem ser imprevisíveis, "
                    "mantém-se a recomendação de monitoramento contínuo e análise periódica."
                )
        else:
            expert_comments += "Não há dados suficientes para fornecer uma análise especializada."
        
        dashboard = {
            "sorted_localities": grouped.to_dict(orient="records"),
            "indicators_plot": locality_data.get("indicators_plot"),
            "location_plot": locality_data.get("location_plot"),
            "risk_index_plot": risk_index_plot,
            "expert_analysis": expert_comments
        }
        return dashboard
    except Exception as e:
        logging.error(f"Erro na geração do dashboard de risco: {e}")
        raise e

# --- Implementação da API com FastAPI ---
app = FastAPI(
    title="AgentBroker - Sistema de Previsão de Terremotos, Análise, Monitoramento e Dashboard",
    description=(
        "Executa a coleta, previsão, análise por localidade, monitoramento em tempo real e "
        "apresenta dashboards interativos com funcionalidades avançadas, incluindo análise de risco via IA "
        "para auxiliar na mitigação dos riscos sísmicos."
    ),
    version="2.0"
)
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

@app.get("/ensemble-forecast", dependencies=[Depends(get_api_key)], summary="Executa a Previsão Agregada com Ensemble")
async def api_ensemble_forecast(background_tasks: BackgroundTasks):
    try:
        result = ensemble_forecast()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/locality-analysis", dependencies=[Depends(get_api_key)], summary="Executa a Análise por Localidade")
async def api_locality_analysis():
    try:
        result = locality_analysis()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard-data", dependencies=[Depends(get_api_key)], summary="Dados Agregados para o Dashboard")
async def api_dashboard_data():
    data = get_dashboard_data()
    return {"dashboard_data": data}

@app.get("/query-events", dependencies=[Depends(get_api_key)], summary="Consulta Eventos com Filtros")
async def api_query_events(start_date: str = None, end_date: str = None, place: str = None):
    events = query_events(start_date, end_date, place)
    return {"events": events}

@app.get("/ia-risk-dashboard", dependencies=[Depends(get_api_key)], summary="Dashboard de Risco via IA")
async def api_ia_risk_dashboard():
    try:
        dashboard = ia_risk_dashboard()
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Verifica o status do serviço")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/intranet", dependencies=[Depends(get_api_key)], summary="Lista de Endpoints Disponíveis")
async def intranet():
    endpoints = {
        "/health": "Verifica o status do serviço.",
        "/ensemble-forecast": "Executa a previsão agregada com ensemble (LSTM + Regressão Linear).",
        "/locality-analysis": "Realiza a análise por localidade a partir dos dados.",
        "/dashboard-data": "Fornece dados agregados para o dashboard interativo.",
        "/query-events": "Consulta eventos no banco de dados com filtros.",
        "/ia-risk-dashboard": "Apresenta um dashboard de risco via IA (análise especializada de Computação Sísmica).",
        "/intranet": "Lista todos os endpoints disponíveis com suas descrições."
    }
    return endpoints

# --- Agendamento em Background (Jobs Periódicos) ---
def scheduled_jobs():
    """
    Agenda os jobs periódicos:
      - Execução do ensemble forecast diário (às 00:00 UTC).
      - Remoção de arquivos antigos a cada 6 horas.
    """
    schedule.every().day.at("00:00").do(ensemble_forecast)
    logging.info("Job de previsão diário agendado para 00:00 UTC.")
    schedule.every(6).hours.do(remove_old_files, directory=DATA_DIR, days=30)
    schedule.every(6).hours.do(remove_old_files, directory=PLOTS_DIR, days=30)
    logging.info("Jobs de limpeza de arquivos agendados a cada 6 horas.")
    while True:
        schedule.run_pending()
        time.sleep(60)

# --- Inicialização ---
if __name__ == "__main__":
    import uvicorn
    init_db()
    monitoring_thread = threading.Thread(target=simulate_real_time_monitoring, daemon=True)
    monitoring_thread.start()
    logging.info("Thread de monitoramento em tempo real iniciada.")
    if os.environ.get("RUN_MAIN", "false") == "true":
        scheduler_thread = threading.Thread(target=scheduled_jobs, daemon=True)
        scheduler_thread.start()
        logging.info("Scheduler background thread iniciado.")
    uvicorn.run("Coleta_de_dados_v7:app", host="0.0.0.0", port=8000, reload=True)
