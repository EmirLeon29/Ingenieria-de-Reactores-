import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuración global de toda la aplicación (solo debe estar en este archivo)
st.set_page_config(
    page_title="Ingenieria de Reactores",
    page_icon="🥼",
    layout="wide",
)

# --- FUNCIONES DE CÁLCULO (reacción de orden 1) ---
def perfil_conversion(es_rtca, n, tau_i, k, paralelo):
    """Devuelve (X acumulada, X por etapa) para cada reactor."""
    if paralelo:
        x = k * tau_i / (1 + k * tau_i) if es_rtca else 1 - np.exp(-k * tau_i)
        return [x] * n, [x] * n

    acumulada, por_etapa = [], []
    c_rel = 1.0  # CA/CA0
    for _ in range(n):
        c_ant = c_rel
        c_rel = c_ant / (1 + k * tau_i) if es_rtca else c_ant * np.exp(-k * tau_i)
        acumulada.append(1 - c_rel)
        por_etapa.append((c_ant - c_rel) / c_ant)
    return acumulada, por_etapa


def x_rtca_serie(tau_total, n, k):
    return 1 - (1 + k * tau_total / n) ** (-n)


def x_rtfp(tau_total, k):
    return 1 - np.exp(-k * tau_total)


# --- PANEL LATERAL ---
st.sidebar.header("Configuración del Sistema")

arreglo = st.sidebar.selectbox(
    "Arreglo de reactores:",
    ["Tanque Individual", "Tanques en Serie", "Tanques en Paralelo"],
)
tipo_reactor = st.sidebar.selectbox(
    "Tipo de reactor:", ["RTCA (Mezcla Completa)", "RTFP (Flujo Pistón)"]
)
es_rtca = "RTCA" in tipo_reactor
paralelo = arreglo == "Tanques en Paralelo"

if arreglo == "Tanque Individual":
    n_tanques = 1
else:
    n_tanques = st.sidebar.slider("Cantidad de tanques (N):", 2, 10, 3)

st.sidebar.markdown("---")
st.sidebar.header("Parámetros Cinéticos y de Operación")

k = st.sidebar.number_input("Constante cinética k (1/min):", min_value=0.001, value=0.25, step=0.05)
ca0 = st.sidebar.number_input(
    "Concentración inicial CA0 (mol/L):",
    min_value=0.01, value=1.0, step=0.1,
    help="En una cinética de primer orden CA0 no afecta la conversión; solo modifica la velocidad de reacción (-rA).",
)

variable_calcular = st.sidebar.radio(
    "¿Qué deseas calcular?",
    ["Porcentaje de Conversión (X)", "Tiempo de Permanencia Total (τ)"],
)

# --- LÓGICA ---
if variable_calcular == "Porcentaje de Conversión (X)":
    tau_total = st.sidebar.number_input("Tiempo de permanencia total τ (min):", min_value=0.1, value=5.0, step=0.5)
    tau_i = tau_total if paralelo else tau_total / n_tanques
else:
    x_deseada_pct = st.sidebar.slider("Conversión deseada (%):", 1.0, 99.0, 80.0)
    x_obj = x_deseada_pct / 100.0
    if es_rtca:
        if paralelo:
            tau_i = x_obj / (k * (1 - x_obj))
            tau_total = tau_i
        else:
            tau_i = (1 / k) * ((1 / (1 - x_obj)) ** (1 / n_tanques) - 1)
            tau_total = tau_i * n_tanques
    else:
        tau_total = -np.log(1 - x_obj) / k
        tau_i = tau_total if paralelo else tau_total / n_tanques

acumulada, por_etapa = perfil_conversion(es_rtca, n_tanques, tau_i, k, paralelo)
x_final = acumulada[-1]

ca_final = ca0 * (1 - x_final)
velocidad_salida = k * ca_final

# --- RESULTADOS ---
col1, col2, col3 = st.columns(3)
col1.metric("Conversión Global (X)", f"{x_final * 100:.2f} %")
col2.metric("Tiempo de Permanencia Total (τ)", f"{tau_total:.2f} min")
col3.metric("Velocidad de Reacción Final (-rA)", f"{velocidad_salida:.4f} mol/(L·min)")

# --- GRÁFICO 1: PERFIL POR REACTOR ---
if paralelo:
    st.subheader("Conversión de cada rama en paralelo")
    st.caption("Las ramas son iguales y operan de forma independiente, por lo que no se acumulan.")
else:
    st.subheader("Perfil de Conversión por Reactor")

etapas = [f"R-{i+1}" for i in range(n_tanques)]
fig, ax = plt.subplots(figsize=(8, 3.5))
if paralelo:
    ax.bar(etapas, [x * 100 for x in acumulada], color="#2b7bba")
    ax.set_ylabel("Conversión de la rama (%)")
else:
    idx = np.arange(n_tanques)
    w = 0.38
    ax.bar(idx - w / 2, [x * 100 for x in acumulada], w, color="#2b7bba", label="Acumulada")
    ax.bar(idx + w / 2, [x * 100 for x in por_etapa], w, color="#e07b39", label="Por etapa")
    ax.set_xticks(idx)
    ax.set_xticklabels(etapas)
    ax.set_ylabel("Conversión (%)")
    ax.legend()
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.5)
st.pyplot(fig)
plt.close(fig)

# --- GRÁFICOS 2 Y 3: COMPARATIVAS ---
st.subheader("Comparativas")
c1, c2 = st.columns(2)

with c1:
    tau_max = max(tau_total * 2, 10 / k * 0.5)
    taus = np.linspace(0.01, tau_max, 300)
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.plot(taus, 100 * x_rtfp(taus, k), label="RTFP", color="#2b7bba")
    ax2.plot(taus, 100 * x_rtca_serie(taus, 1, k), label="RTCA (N=1)", color="#999999")
    if n_tanques > 1 and not paralelo:
        ax2.plot(taus, 100 * x_rtca_serie(taus, n_tanques, k),
                 label=f"RTCA en serie (N={n_tanques})", color="#e07b39")
    ax2.scatter([tau_total], [100 * x_final], color="red", zorder=5, label="Punto actual")
    ax2.set_xlabel("τ total (min)")
    ax2.set_ylabel("Conversión (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("X vs. τ")
    ax2.grid(linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)
    plt.close(fig2)

with c2:
    ns = np.arange(1, 11)
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    ax3.plot(ns, [100 * x_rtca_serie(tau_total, n, k) for n in ns], "o-",
             color="#e07b39", label="RTCA en serie")
    ax3.axhline(100 * x_rtfp(tau_total, k), color="#2b7bba", linestyle="--", label="RTFP (límite N→∞)")
    ax3.set_xlabel("Número de tanques (N)")
    ax3.set_ylabel("Conversión (%)")
    ax3.set_title(f"X vs. N (τ total = {tau_total:.2f} min)")
    ax3.set_xticks(ns)
    ax3.grid(linestyle="--", alpha=0.5)
    ax3.legend(fontsize=8)
    st.pyplot(fig3)
    plt.close(fig3)

