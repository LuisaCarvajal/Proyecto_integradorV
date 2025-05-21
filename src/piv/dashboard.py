import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns

# --- Carga de datos ---
df = pd.read_csv("src/piv/static/data/historical.csv")
df.columns = df.columns.str.strip().str.replace('\ufeff', '')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- Cálculos adicionales ---
df['MA20'] = df['close_samsung'].rolling(window=20).mean()
df['MA50'] = df['close_samsung'].rolling(window=50).mean()

# RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['close_samsung'])

# --- Interfaz Streamlit ---
st.title("📊 Dashboard Financiero: Samsung vs KOSPI")

# KPIs
st.subheader("📌 Indicadores Clave")
st.metric("Último precio Samsung", f"₩{df['close_samsung'].iloc[-1]:,.0f}")
st.metric("Retorno Diario Último", f"{df['daily_return'].iloc[-1]*100:.2f}%")
st.metric("Volumen Último Día", f"{df['volume_005930.ks'].iloc[-1]:,.0f}")

# Selección de módulos
option = st.selectbox("Selecciona una sección:", [
    "Media Móvil",
    "RSI",
    "Predicción ARIMA",
    "Predicción SARIMA",
    "Volatilidad"

])

if option == "Media Móvil":
    st.subheader("📈 Media Móvil de Samsung")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['close_samsung'], label='Cierre Samsung', color='blue')
    ax.plot(df.index, df['MA20'], label='MA 20 días', color='orange')
    ax.plot(df.index, df['MA50'], label='MA 50 días', color='green')
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.set_title("Precio de Cierre con Media Móvil (Samsung)")
    ax.legend()
    st.pyplot(fig)

elif option == "Volatilidad":
    # Volatilidad de Samsung
    st.subheader('Volatilidad 20 días - Samsung')
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=df.index, y=df['volatility_20'], mode='lines', name='Volatilidad 20D'))
    fig_vol.update_layout(title='Volatilidad móvil de 20 días (Samsung)',
                xaxis_title='Fecha', yaxis_title='Volatilidad',template='plotly_white')
    st.plotly_chart(fig_vol)

elif option == "RSI":
    st.subheader("📉 RSI (Relative Strength Index)")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax.axhline(70, color='red', linestyle='--', label='Sobrecomprado')
    ax.axhline(30, color='green', linestyle='--', label='Sobrevendido')
    ax.set_title("Índice de Fuerza Relativa - RSI")
    ax.legend()
    st.pyplot(fig)

elif option == "Predicción ARIMA":
    st.subheader("📊 Predicción con ARIMA")
    series = df['close_samsung'].dropna()
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    st.line_chart(pd.concat([series, forecast], axis=0, ignore_index=True))
    st.write("**Próximos 10 días de predicción ARIMA:**")
    st.write(forecast)

elif option == "Predicción SARIMA":
    st.subheader("🔮 Predicción con SARIMA")
    sarima_series = df['close_samsung'].dropna()
    sarima_model = SARIMAX(sarima_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(steps=10)
    st.line_chart(pd.concat([sarima_series, sarima_forecast], axis=0, ignore_index=True))
    st.write("**Próximos 10 días de predicción SARIMA:**")
    st.write(sarima_forecast)

st.markdown("---")

