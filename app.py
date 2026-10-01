"""
Interfaz web del laboratorio (Streamlit).

Permite elegir el ejercicio (Dólar, Glucosa o Energía), ingresar por teclado
los valores de las variables independientes y ver la predicción del modelo
exportado en models/*.joblib.

Ejecutar:
    streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

from config import DIR_DATOS, DIR_FIGURAS, DIR_MODELOS, EJERCICIOS

st.set_page_config(page_title="Laboratorio Minería de Datos", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------
# Carga de recursos (se cachea para no leer el disco en cada interacción)
# --------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(nombre_archivo):
    return joblib.load(DIR_MODELOS / nombre_archivo)


@st.cache_data
def rangos_entrenamiento(nombre_csv, variables):
    """Mínimo y máximo de cada variable en los datos de entrenamiento."""
    df = pd.read_csv(DIR_DATOS / nombre_csv)
    return {v: (float(df[v].min()), float(df[v].max())) for v in variables}


def crear_entrada(variable, cfg):
    """Dibuja el campo numérico de una variable según los rangos de config.py."""
    minimo, maximo, defecto, paso = cfg["rangos"][variable]
    etiqueta = cfg["etiquetas"][variable]
    es_entero = all(isinstance(v, int) for v in (minimo, maximo, defecto, paso))

    if es_entero:
        return st.number_input(etiqueta, min_value=minimo, max_value=maximo,
                               value=defecto, step=paso, key=f"{variable}_{cfg['csv']}")
    return st.number_input(etiqueta, min_value=float(minimo), max_value=float(maximo),
                           value=float(defecto), step=float(paso), format="%.4f"
                           if paso < 0.01 else "%.2f", key=f"{variable}_{cfg['csv']}")


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------
st.title("📈 Predicciones con regresión lineal múltiple")
st.caption("Laboratorio 1 · Minería de Datos · metodología CRISP-DM")

with st.sidebar:
    st.header("Ejercicio")
    clave = st.radio(
        "Seleccione el modelo",
        options=list(EJERCICIOS.keys()),
        format_func=lambda k: EJERCICIOS[k]["nombre"],
    )
    st.divider()
    st.write("Ingrese los valores en el formulario y presione **Predecir**.")

cfg = EJERCICIOS[clave]
ruta_modelo = DIR_MODELOS / cfg["modelo"]
if not ruta_modelo.exists():
    st.error("No se encontró el modelo exportado. Ejecute primero `python entrenamiento.py`.")
    st.stop()

paquete = cargar_modelo(cfg["modelo"])
modelo = paquete["modelo"]
limites = rangos_entrenamiento(cfg["csv"], tuple(cfg["variables"]))

st.subheader(cfg["nombre"])
tab_pred, tab_modelo, tab_graficas = st.tabs(["Predicción", "Modelo", "Gráficas"])

# ---- Pestaña 1: formulario de predicción ----
with tab_pred:
    columnas = st.columns(len(cfg["variables"]))
    valores = {}
    for col, variable in zip(columnas, cfg["variables"]):
        with col:
            valores[variable] = crear_entrada(variable, cfg)

    if st.button("Predecir", type="primary"):
        entrada = pd.DataFrame([valores], columns=paquete["variables"])
        resultado = float(modelo.predict(entrada)[0])

        st.metric(label=f"{paquete['objetivo']} estimado", value=f"{resultado:,.2f} {cfg['unidad']}")

        # aviso si el usuario se sale de lo que el modelo vio al entrenarse
        fuera = [v for v, (lo, hi) in limites.items() if not lo <= valores[v] <= hi]
        if fuera:
            st.warning(
                "Estos valores están fuera del rango de los datos de entrenamiento, "
                "así que la predicción es una extrapolación: " + ", ".join(fuera)
            )

# ---- Pestaña 2: información del modelo ----
with tab_modelo:
    met = paquete["metricas_prueba"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (prueba)", f"{met['R2']:.4f}")
    c2.metric("MSE (prueba)", f"{met['MSE']:,.2f}")
    c3.metric("RMSE (prueba)", f"{met['RMSE']:,.2f} {cfg['unidad']}")

    tabla = pd.DataFrame({
        "Variable": ["Intercepto"] + paquete["variables"],
        "Coeficiente": [modelo.intercept_] + list(modelo.coef_),
    })
    st.write("**Ecuación del modelo**")
    st.dataframe(tabla, hide_index=True)

# ---- Pestaña 3: gráficas generadas en el entrenamiento ----
with tab_graficas:
    imagenes = [DIR_FIGURAS / f"{clave}_dispersion_{v}.png" for v in cfg["variables"]]
    imagenes += [DIR_FIGURAS / f"{clave}_importancia.png",
                 DIR_FIGURAS / f"{clave}_real_vs_predicho.png"]
    for ruta in imagenes:
        if ruta.exists():
            st.image(str(ruta))
