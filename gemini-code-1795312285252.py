import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador de Reactores", layout="wide")
st.title("Simulador de Reactores Ideales: RTCA y RTFP")

# --- PANEL LATERAL DE CONFIGURACIÓN ---
st.sidebar.header("Configuración del Sistema")

arreglo = st.sidebar.selectbox(
    "Arreglo de reactores:",
    ["Tanque Individual", "Tanques en Serie", "Tanques en Paralelo"]
)

tipo_reactor = st.sidebar.selectbox(
    "Tipo de reactor:",
    ["RTCA (Mezcla Completa)", "RTFP (Flujo Pistón)"]
)

if arreglo == "Tanque Individual":
    n_tanques = 1
else:
    n_tanques = st.sidebar.slider("Cantidad de tanques (N):", min_value=2, max_value=10, value=3)

st.sidebar.markdown("---")
st.sidebar.header("Parámetros Cinéticos y de Operación")

k = st.sidebar.number_input("Constante cinética k (1/min):", min_value=0.001, value=0.25, step=0.05)
ca0 = st.sidebar.number_input("Concentración inicial CA0 (mol/L):", min_value=0.01, value=1.0, step=0.1)

variable_calcular = st.sidebar.radio(
    "¿Qué deseas calcular?",
    ["Porcentaje de Conversión (X)", "Tiempo de Permanencia Total (τ)"]
)

# --- LÓGICA DE CÁLCULO (Orden 1) ---
if variable_calcular == "Porcentaje de Conversión (X)":
    tau_total = st.sidebar.number_input("Tiempo de permanencia total τ (min):", min_value=0.1, value=5.0, step=0.5)
    
    if arreglo == "Tanques en Paralelo":
        # El flujo se divide entre N ramas iguales; cada rama tiene el mismo tau que el sistema total
        tau_i = tau_total
        if "RTCA" in tipo_reactor:
            x_final = (k * tau_i) / (1 + k * tau_i)
        else:
            x_final = 1 - np.exp(-k * tau_i)
        conversiones_etapa = [x_final] * n_tanques
    else:
        # Individual o en Serie
        tau_i = tau_total / n_tanques
        conversiones_etapa = []
        ca_actual = ca0
        for _ in range(n_tanques):
            if "RTCA" in tipo_reactor:
                ca_salida = ca_actual / (1 + k * tau_i)
            else:
                ca_salida = ca_actual * np.exp(-k * tau_i)
            x_etapa = (ca0 - ca_salida) / ca0
            conversiones_etapa.append(x_etapa)
            ca_actual = ca_salida
        x_final = conversiones_etapa[-1]

else:
    x_deseada_pct = st.sidebar.slider("Conversión deseada (%):", min_value=1.0, max_value=99.0, value=80.0)
    x_final = x_deseada_pct / 100.0
    
    if "RTCA" in tipo_reactor:
        if arreglo == "Tanques en Paralelo":
            tau_total = x_final / (k * (1 - x_final))
        else:
            tau_i = (1 / k) * ((1 / (1 - x_final))**(1 / n_tanques) - 1)
            tau_total = tau_i * n_tanques
    else:
        tau_total = -np.log(1 - x_final) / k
        tau_i = tau_total / n_tanques if arreglo != "Tanques en Paralelo" else tau_total

    # Calcular perfil por etapa para graficar
    conversiones_etapa = []
    ca_actual = ca0
    for _ in range(n_tanques):
        if arreglo == "Tanques en Paralelo":
            conversiones_etapa.append(x_final)
        else:
            if "RTCA" in tipo_reactor:
                ca_salida = ca_actual / (1 + k * tau_i)
            else:
                ca_salida = ca_actual * np.exp(-k * tau_i)
            conversiones_etapa.append((ca0 - ca_salida) / ca0)
            ca_actual = ca_salida

# Velocidad de reacción a la salida (-rA = k * CA_final)
ca_final = ca0 * (1 - x_final)
velocidad_salida = k * ca_final

# --- RESULTADOS EN PANTALLA ---
col1, col2, col3 = st.columns(3)
col1.metric("Conversión Global (X)", f"{x_final * 100:.2f} %")
col2.metric("Tiempo de Permanencia Total (τ)", f"{tau_total:.2f} min")
col3.metric("Velocidad de Reacción Final (-rA)", f"{velocidad_salida:.4f} mol/(L·min)")

# --- GRÁFICO ---
st.subheader("Perfil de Conversión por Reactor")
fig, ax = plt.subplots(figsize=(8, 3.5))
etapas = [f"R-{i+1}" for i in range(n_tanques)]
ax.bar(etapas, [x * 100 for x in conversiones_etapa], color="#2b7bba")
ax.set_ylabel("Conversión Acumulada (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.5)
st.pyplot(fig)