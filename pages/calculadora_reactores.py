import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Calculadora de Reactores", layout="wide")

# =====================================================================
# CONSTANTES Y CATÁLOGOS
# =====================================================================
R_ATM = 0.082057   # L·atm/(mol·K)
R_J = 8.314        # J/(mol·K)

ORDENES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ARREGLOS = ["Tanque Individual", "Tanques en Serie", "Tanques en Paralelo"]
TIPOS = ["RTCA (Mezcla Completa)", "RTFP (Flujo Pistón)"]
FASES = ["Líquida", "Gas ideal"]
OBJETIVOS = ["Conversión (X)", "Tiempo de permanencia total (τ)"]
DATOS_CONOCIDOS = ["τ total (min)", "Volumen total V (L)"]

# factores a unidades base: atm | L/min | mol/min | g/min
PRES = {"atm": 1.0, "bar": 1 / 1.01325, "kPa": 1 / 101.325, "psi": 1 / 14.696}
FLUJOS = {
    "Volumétrico": {"L/min": 1.0, "L/h": 1 / 60, "m³/h": 1000 / 60, "m³/min": 1000.0},
    "Molar (de A)": {"mol/min": 1.0, "mol/h": 1 / 60, "kmol/h": 1000 / 60, "kmol/min": 1000.0},
    "Másico (de A)": {"g/min": 1.0, "g/h": 1 / 60, "kg/h": 1000 / 60, "kg/min": 1000.0},
}

DEFAULTS = dict(
    arreglo="Tanque Individual", tipo=TIPOS[0], n_tanques=3,
    orden=1.0, k_ref=0.25, usar_arr=False, ea=50.0, t_ref=25.0,
    fase="Líquida", ca0=1.0, y_a0=1.0,
    temp=25.0, t_unit="°C", pres=1.0, p_unit="atm",
    flujo_tipo="Volumétrico", flujo_unit="L/min", flujo_val=10.0, mw=60.0,
    objetivo=OBJETIVOS[0], dato=DATOS_CONOCIDOS[0], tau=5.0, vol=50.0, x_pct=80.0,
)


def k_unidades(n):
    if n == 0:
        return "mol/(L·min)"
    if n == 1:
        return "1/min"
    if n == 2:
        return "L/(mol·min)"
    return f"(mol/L)^{1 - n:g} · 1/min"


def idx(opciones, valor):
    return opciones.index(valor) if valor in opciones else 0


# =====================================================================
# MOTOR DE CÁLCULO  (-rA = k·CA^n, densidad constante, isotérmico)
# =====================================================================
def ca_cstr(ca_in, tau, k, n):
    """Concentración de salida de un RTCA: CA_in - CA = k·τ·CA^n (bisección)."""
    ca_in, tau = np.broadcast_arrays(np.asarray(ca_in, float), np.asarray(tau, float))
    if n == 0:
        return np.maximum(ca_in - k * tau, 0.0)
    lo = np.zeros(ca_in.shape)
    hi = ca_in.copy()
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        f = ca_in - mid - k * tau * mid ** n   # decreciente en mid
        pos = f > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def ca_pfr(ca_in, tau, k, n):
    """Concentración de salida de un RTFP: solución analítica de dC/dτ = -k·C^n."""
    ca_in, tau = np.broadcast_arrays(np.asarray(ca_in, float), np.asarray(tau, float))
    if n == 1:
        return ca_in * np.exp(-k * tau)
    with np.errstate(all="ignore"):
        base = ca_in ** (1 - n) + (n - 1) * k * tau
        return np.where(base > 0, base ** (1 / (1 - n)), 0.0)


def conversion_desde_tau(es_rtca, paralelo, n_et, ca0, tau_total, k, n):
    tau_total = np.asarray(tau_total, float)
    etapas = 1 if paralelo else n_et
    tau_i = tau_total if paralelo else tau_total / n_et
    ca = np.full(tau_total.shape, float(ca0))
    for _ in range(etapas):
        ca = ca_cstr(ca, tau_i, k, n) if es_rtca else ca_pfr(ca, tau_i, k, n)
    return 1 - ca / ca0


def tau_requerida(es_rtca, paralelo, n_et, ca0, x, k, n):
    ca_t = ca0 * (1 - x)
    if not es_rtca:
        if n == 1:
            return np.log(ca0 / ca_t) / k
        return (ca0 ** (1 - n) - ca_t ** (1 - n)) / ((1 - n) * k)
    tau_unico = (ca0 - ca_t) / (k * ca_t ** n)
    if paralelo or n_et == 1:
        return float(tau_unico)
    lo, hi = 0.0, float(tau_unico)   # el total en serie nunca supera al de un solo tanque
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(conversion_desde_tau(True, False, n_et, ca0, mid, k, n)) < x:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def perfil(es_rtca, paralelo, n_et, ca0, tau_total, k, n):
    """Lista de (CA_entrada, CA_salida, τ_i) por reactor."""
    tau_i = tau_total if paralelo else tau_total / n_et
    paso = (lambda c: float(ca_cstr(c, tau_i, k, n))) if es_rtca else (lambda c: float(ca_pfr(c, tau_i, k, n)))
    filas, ca = [], ca0
    if paralelo:
        c_out = paso(ca0)
        return [(ca0, c_out, tau_i)] * n_et
    for _ in range(n_et):
        c_out = paso(ca)
        filas.append((ca, c_out, tau_i))
        ca = c_out
    return filas


def preparar(p):
    if p["t_unit"] == "°C":
        T, Tref = p["temp"] + 273.15, p["t_ref"] + 273.15
    else:
        T, Tref = p["temp"], p["t_ref"]
    P_atm = p["pres"] * PRES[p["p_unit"]]

    k = p["k_ref"]
    if p["usar_arr"]:
        k = k * np.exp(-p["ea"] * 1000 / R_J * (1 / T - 1 / Tref))

    ca0 = p["ca0"] if p["fase"] == "Líquida" else p["y_a0"] * P_atm / (R_ATM * T)

    val = p["flujo_val"] * FLUJOS[p["flujo_tipo"]][p["flujo_unit"]]
    if p["flujo_tipo"] == "Volumétrico":
        Q = val
        FA0 = ca0 * Q
    elif p["flujo_tipo"] == "Molar (de A)":
        FA0 = val
        Q = FA0 / ca0
    else:
        FA0 = val / p["mw"]
        Q = FA0 / ca0
    return dict(T=T, P_atm=P_atm, k=k, ca0=ca0, Q=Q, FA0=FA0)


def calcular(p):
    d = preparar(p)
    es_rtca = "RTCA" in p["tipo"]
    paralelo = p["arreglo"] == "Tanques en Paralelo"
    N = 1 if p["arreglo"] == "Tanque Individual" else p["n_tanques"]
    n, k, ca0 = p["orden"], d["k"], d["ca0"]

    if p["objetivo"] == OBJETIVOS[0]:
        tau = p["tau"] if p["dato"] == DATOS_CONOCIDOS[0] else p["vol"] / d["Q"]
    else:
        tau = tau_requerida(es_rtca, paralelo, N, ca0, p["x_pct"] / 100, k, n)

    filas = perfil(es_rtca, paralelo, N, ca0, tau, k, n)
    ca_final = filas[-1][1]
    x_final = 1 - ca_final / ca0
    V_total = tau * d["Q"]   # en paralelo: N ramas con Q/N y τ cada una → mismo V total
    d.update(es_rtca=es_rtca, paralelo=paralelo, N=N, n=n, tau=tau, filas=filas,
             ca_final=ca_final, x=x_final, V=V_total, V_i=V_total / N,
             FA_out=d["FA0"] * (1 - x_final), rA_out=k * ca_final ** n)
    return d


# =====================================================================
# PÁGINA 1: CAPTURA DE DATOS
# =====================================================================
def pagina_entrada():
    p = st.session_state.get("params", DEFAULTS)
    st.title("Calculadora de Reactores Ideales: RTCA y RTFP")
    st.caption("Paso 1 de 2 · Captura los datos del sistema y presiona **Calcular**.")

    c1, c2, c3 = st.columns(3)

    # ---- 1. Sistema
    with c1:
        st.subheader("1. Sistema de reactores")
        arreglo = st.selectbox("Arreglo de reactores:", ARREGLOS, index=idx(ARREGLOS, p["arreglo"]))
        tipo = st.selectbox("Tipo de reactor:", TIPOS, index=idx(TIPOS, p["tipo"]))
        if arreglo == "Tanque Individual":
            n_tanques = 1
        else:
            n_tanques = st.slider("Cantidad de reactores (N):", 2, 10, max(2, int(p["n_tanques"])))
        st.caption("Los N reactores son idénticos (mismo volumen).")

    # ---- 2. Operación
    with c2:
        st.subheader("2. Condiciones de operación")
        a, b = st.columns([2, 1])
        t_unit = b.selectbox("Unidad T", ["°C", "K"], index=idx(["°C", "K"], p["t_unit"]))
        temp = a.number_input(f"Temperatura ({t_unit}):", value=float(p["temp"]), step=5.0)
        a, b = st.columns([2, 1])
        p_unit = b.selectbox("Unidad P", list(PRES), index=idx(list(PRES), p["p_unit"]))
        pres = a.number_input(f"Presión ({p_unit}):", min_value=0.001, value=float(p["pres"]), step=0.5)

        flujo_tipo = st.selectbox("Tipo de flujo de alimentación:", list(FLUJOS),
                                  index=idx(list(FLUJOS), p["flujo_tipo"]))
        unidades = list(FLUJOS[flujo_tipo])
        a, b = st.columns([2, 1])
        flujo_unit = b.selectbox("Unidad flujo", unidades, index=idx(unidades, p["flujo_unit"]))
        flujo_val = a.number_input(f"Flujo ({flujo_unit}):", min_value=0.0001,
                                   value=float(p["flujo_val"]), step=1.0)
        mw = float(p["mw"])
        if flujo_tipo == "Másico (de A)":
            mw = st.number_input("Masa molar de A (g/mol):", min_value=0.01, value=mw, step=1.0)

    # ---- 3. Cinética y alimentación
    with c3:
        st.subheader("3. Cinética y alimentación")
        orden = st.selectbox("Orden de reacción (n):", ORDENES, index=idx(ORDENES, p["orden"]),
                             format_func=lambda v: f"{v:g}")
        st.caption("Reacción A → productos, con  −rA = k·CA^n")
        k_ref = st.number_input(f"Constante cinética k ({k_unidades(orden)}):",
                                min_value=0.0001, value=float(p["k_ref"]), step=0.05, format="%.4f")
        usar_arr = st.checkbox("Corregir k con la temperatura (Arrhenius)", value=p["usar_arr"])
        ea, t_ref = float(p["ea"]), float(p["t_ref"])
        if usar_arr:
            ea = st.number_input("Energía de activación Ea (kJ/mol):", min_value=0.0, value=ea, step=5.0)
            t_ref = st.number_input(f"Temperatura a la que se midió k ({t_unit}):", value=t_ref, step=5.0)

        fase = st.radio("Fase de la alimentación:", FASES, index=idx(FASES, p["fase"]), horizontal=True)
        ca0, y_a0 = float(p["ca0"]), float(p["y_a0"])
        if fase == "Líquida":
            ca0 = st.number_input("Concentración inicial CA0 (mol/L):", min_value=0.0001, value=ca0, step=0.1)
            st.caption("En fase líquida la presión no entra al cálculo.")
        else:
            y_a0 = st.number_input("Fracción molar de A en la alimentación (yA0):",
                                   min_value=0.001, max_value=1.0, value=y_a0, step=0.05)
            T_K = temp + 273.15 if t_unit == "°C" else temp
            if T_K > 0:
                ca0_calc = y_a0 * pres * PRES[p_unit] / (R_ATM * T_K)
                st.caption(f"CA0 = yA0·P/(R·T) = **{ca0_calc:.4f} mol/L**")

    st.markdown("---")
    st.subheader("4. ¿Qué deseas calcular?")
    objetivo = st.radio("Objetivo:", OBJETIVOS, index=idx(OBJETIVOS, p["objetivo"]), horizontal=True)
    dato, tau, vol, x_pct = p["dato"], float(p["tau"]), float(p["vol"]), float(p["x_pct"])
    if objetivo == OBJETIVOS[0]:
        dato = st.radio("Dato conocido:", DATOS_CONOCIDOS, index=idx(DATOS_CONOCIDOS, p["dato"]), horizontal=True)
        if dato == DATOS_CONOCIDOS[0]:
            tau = st.number_input("Tiempo de permanencia total τ (min):", min_value=0.1, value=tau, step=0.5)
        else:
            vol = st.number_input("Volumen total V (L):", min_value=0.1, value=vol, step=5.0)
    else:
        x_pct = st.slider("Conversión deseada (%):", 1.0, 99.0, x_pct)

    st.info("Supuestos: isotérmico, densidad constante (sin cambio de volumen por reacción) "
            "y reactores ideales. En fase gas, CA0 se obtiene con gas ideal.")

    if st.button("Calcular ➜", type="primary"):
        st.session_state["params"] = dict(
            arreglo=arreglo, tipo=tipo, n_tanques=n_tanques, orden=orden, k_ref=k_ref,
            usar_arr=usar_arr, ea=ea, t_ref=t_ref, fase=fase, ca0=ca0, y_a0=y_a0,
            temp=temp, t_unit=t_unit, pres=pres, p_unit=p_unit,
            flujo_tipo=flujo_tipo, flujo_unit=flujo_unit, flujo_val=flujo_val, mw=mw,
            objetivo=objetivo, dato=dato, tau=tau, vol=vol, x_pct=x_pct,
        )
        st.session_state["pagina"] = "resultados"
        st.rerun()


# =====================================================================
# PÁGINA 2: RESULTADOS
# =====================================================================
def pagina_resultados():
    p = st.session_state["params"]
    r = calcular(p)
    n, k, ca0 = r["n"], r["k"], r["ca0"]

    top1, top2 = st.columns([5, 1])
    top1.title("Resultados")
    top1.caption(f"{p['arreglo']} · {p['tipo']} · N = {r['N']} · orden n = {n:g}")
    if top2.button("⬅ Modificar datos"):
        st.session_state["pagina"] = "entrada"
        st.rerun()

    if n < 1 and r["x"] > 0.9999:
        st.info("Con orden < 1 la reacción puede consumir todo el reactivo en un tiempo finito: "
                "el τ o el volumen dados alcanzan o superan la conversión completa.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Conversión global (X)", f"{r['x'] * 100:.2f} %")
    m2.metric("Tiempo de permanencia total (τ)", f"{r['tau']:.3f} min")
    m3.metric("Volumen total (V)", f"{r['V']:.2f} L", f"{r['V'] / 1000:.4f} m³", delta_color="off")

    m4, m5, m6 = st.columns(3)
    m4.metric("Volumen por reactor", f"{r['V_i']:.2f} L")
    m5.metric("CA a la salida", f"{r['ca_final']:.4f} mol/L")
    m6.metric("Velocidad final (−rA)", f"{r['rA_out']:.4g} mol/(L·min)")

    st.subheader("Balance y condiciones de alimentación")
    resumen = {
        "k a la temperatura de operación": f"{k:.4g} {k_unidades(n)}",
        "T de operación": f"{r['T']:.2f} K ({r['T'] - 273.15:.2f} °C)",
        "P de operación": f"{r['P_atm']:.3f} atm",
        "CA0": f"{ca0:.4f} mol/L",
        "Flujo volumétrico Q": f"{r['Q']:.4g} L/min",
        "Flujo molar de A a la entrada (FA0)": f"{r['FA0']:.4g} mol/min",
        "Flujo molar de A a la salida (FA)": f"{r['FA_out']:.4g} mol/min",
        "A convertido": f"{r['FA0'] - r['FA_out']:.4g} mol/min",
    }
    if p["flujo_tipo"] == "Másico (de A)":
        resumen["Flujo másico de A sin reaccionar"] = f"{r['FA_out'] * p['mw']:.4g} g/min"
    st.table(pd.DataFrame({"Valor": resumen}))

    st.subheader("Detalle por reactor")
    filas = []
    for i, (c_in, c_out, tau_i) in enumerate(r["filas"], start=1):
        filas.append({
            "Reactor": f"R-{i}",
            "τ (min)": round(tau_i, 4),
            "V (L)": round(tau_i * (r["Q"] / (r["N"] if r["paralelo"] else 1)), 3),
            "CA entrada (mol/L)": round(c_in, 5),
            "CA salida (mol/L)": round(c_out, 5),
            "X acumulada (%)": round(100 * (1 - c_out / ca0), 2),
            "X por etapa (%)": round(100 * (1 - c_out / c_in), 2) if c_in > 0 else 0.0,
        })
    df = pd.DataFrame(filas)
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("Descargar tabla (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "detalle_reactores.csv", "text/csv")

    # ---- Gráfico 1
    if r["paralelo"]:
        st.subheader("Conversión de cada rama en paralelo")
        st.caption("Las ramas son iguales y operan de forma independiente, por lo que no se acumulan.")
    else:
        st.subheader("Perfil de conversión por reactor")
    xa = [100 * (1 - f[1] / ca0) for f in r["filas"]]
    xe = [100 * (1 - f[1] / f[0]) if f[0] > 0 else 0 for f in r["filas"]]
    etiquetas = [f"R-{i + 1}" for i in range(r["N"])]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if r["paralelo"]:
        ax.bar(etiquetas, xa, color="#2b7bba")
        ax.set_ylabel("Conversión de la rama (%)")
    else:
        ii, w = np.arange(r["N"]), 0.38
        ax.bar(ii - w / 2, xa, w, color="#2b7bba", label="Acumulada")
        ax.bar(ii + w / 2, xe, w, color="#e07b39", label="Por etapa")
        ax.set_xticks(ii)
        ax.set_xticklabels(etiquetas)
        ax.set_ylabel("Conversión (%)")
        ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

    # ---- Gráficos comparativos
    st.subheader("Comparativas")
    g1, g2 = st.columns(2)
    N = r["N"]

    with g1:
        taus = np.linspace(0.001, 2 * r["tau"], 200)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        ax2.plot(taus, 100 * conversion_desde_tau(False, False, 1, ca0, taus, k, n), label="RTFP", color="#2b7bba")
        ax2.plot(taus, 100 * conversion_desde_tau(True, False, 1, ca0, taus, k, n), label="RTCA (N=1)", color="#999999")
        if N > 1 and not r["paralelo"]:
            ax2.plot(taus, 100 * conversion_desde_tau(True, False, N, ca0, taus, k, n),
                     label=f"RTCA en serie (N={N})", color="#e07b39")
        ax2.scatter([r["tau"]], [100 * r["x"]], color="red", zorder=5, label="Punto actual")
        ax2.set_xlabel("τ total (min)")
        ax2.set_ylabel("Conversión (%)")
        ax2.set_ylim(0, 105)
        ax2.set_title(f"X vs. τ (n = {n:g})")
        ax2.grid(linestyle="--", alpha=0.5)
        ax2.legend(fontsize=8)
        st.pyplot(fig2)
        plt.close(fig2)

    with g2:
        ns = np.arange(1, 11)
        x_ns = [100 * float(conversion_desde_tau(True, False, int(i), ca0, r["tau"], k, n)) for i in ns]
        x_pfr = 100 * float(conversion_desde_tau(False, False, 1, ca0, r["tau"], k, n))
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.plot(ns, x_ns, "o-", color="#e07b39", label="RTCA en serie")
        ax3.axhline(x_pfr, color="#2b7bba", linestyle="--", label="RTFP (límite N→∞)")
        ax3.set_xlabel("Número de tanques (N)")
        ax3.set_ylabel("Conversión (%)")
        ax3.set_title(f"X vs. N (τ total = {r['tau']:.2f} min)")
        ax3.set_xticks(ns)
        ax3.grid(linestyle="--", alpha=0.5)
        ax3.legend(fontsize=8)
        st.pyplot(fig3)
        plt.close(fig3)


# =====================================================================
# NAVEGACIÓN
# =====================================================================
if "pagina" not in st.session_state:
    st.session_state["pagina"] = "entrada"

if st.session_state["pagina"] == "entrada":
    pagina_entrada()
else:
    pagina_resultados()
