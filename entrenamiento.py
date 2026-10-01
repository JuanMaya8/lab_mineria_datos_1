"""
Entrenamiento de los tres modelos de regresión lineal múltiple del laboratorio.

Sigue las fases de CRISP-DM:
    1. Comprensión de los datos   -> explorar() y gráfica de correlación
    2. Preparación de los datos   -> cargar_datos() y división entrenamiento/prueba
    3. Modelado                   -> ajustar_modelo()
    4. Evaluación                 -> evaluar() y analizar_coeficientes()
    5. Despliegue                 -> exportar_modelo() (joblib) + app.py (Streamlit)

Uso:
    python entrenamiento.py
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo guarda las imágenes en disco
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (DIR_DATOS, DIR_FIGURAS, DIR_MODELOS, DIR_RESULTADOS,
                    EJERCICIOS, SEMILLA, TAMANO_PRUEBA)

COLOR_PUNTOS = "#3b82c4"
COLOR_RECTA = "#d9480f"


# --------------------------------------------------------------------------
# 1 y 2. Comprensión y preparación de los datos
# --------------------------------------------------------------------------
def cargar_datos(cfg):
    """Lee el CSV y verifica que no haya nulos, duplicados ni columnas faltantes."""
    df = pd.read_csv(DIR_DATOS / cfg["csv"])
    columnas = cfg["variables"] + [cfg["objetivo"]]
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {cfg['csv']}: {faltantes}")
    return df[columnas]


def explorar(df):
    """Resumen de calidad de datos que se usa luego en el informe."""
    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nulos": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "estadisticas": df.describe().round(4).to_dict(),
    }


# --------------------------------------------------------------------------
# 3. Modelado
# --------------------------------------------------------------------------
def ajustar_modelo(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo


# --------------------------------------------------------------------------
# 4. Evaluación
# --------------------------------------------------------------------------
def evaluar(modelo, X, y):
    pred = modelo.predict(X)
    mse = mean_squared_error(y, pred)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }


def analizar_coeficientes(modelo, X_train, y_train, variables):
    """
    Devuelve, por variable: coeficiente, error estándar, estadístico t, p-valor
    y coeficiente estandarizado (beta). El beta permite comparar la importancia
    de variables que están en escalas distintas.
    """
    n, k = X_train.shape
    X_ext = np.column_stack([np.ones(n), X_train.values])
    coefs = np.concatenate([[modelo.intercept_], modelo.coef_])

    residuos = y_train.values - X_ext @ coefs
    sigma2 = residuos @ residuos / (n - k - 1)
    cov = sigma2 * np.linalg.inv(X_ext.T @ X_ext)
    error_est = np.sqrt(np.diag(cov))
    t = coefs / error_est
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k - 1))

    tabla = [{
        "variable": "Intercepto",
        "coeficiente": float(coefs[0]),
        "error_estandar": float(error_est[0]),
        "t": float(t[0]),
        "p_valor": float(p[0]),
        "beta_estandarizado": None,
    }]
    for i, var in enumerate(variables, start=1):
        beta = coefs[i] * X_train[var].std() / y_train.std()
        tabla.append({
            "variable": var,
            "coeficiente": float(coefs[i]),
            "error_estandar": float(error_est[i]),
            "t": float(t[i]),
            "p_valor": float(p[i]),
            "beta_estandarizado": float(beta),
        })
    return tabla


# --------------------------------------------------------------------------
# Visualizaciones
# --------------------------------------------------------------------------
def graficar_dispersion(df, cfg, clave):
    """Una gráfica por variable independiente contra la variable dependiente."""
    y = df[cfg["objetivo"]]
    for var in cfg["variables"]:
        x = df[var]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter(x, y, s=10, alpha=0.35, color=COLOR_PUNTOS)
        pendiente, corte = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, pendiente * xs + corte, color=COLOR_RECTA, linewidth=2,
                label="Recta de ajuste simple")
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{var} vs {cfg['objetivo']}  (r = {r:.3f})")
        ax.set_xlabel(var)
        ax.set_ylabel(f"{cfg['objetivo']} ({cfg['unidad']})")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(DIR_FIGURAS / f"{clave}_dispersion_{var}.png", dpi=150)
        plt.close(fig)


def graficar_correlacion(df, cfg, clave):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            color_texto = "white" if abs(corr.iloc[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color_texto)
    ax.set_title(f"Matriz de correlación - {cfg['nombre']}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"{clave}_correlacion.png", dpi=150)
    plt.close(fig)


def graficar_real_vs_predicho(y_real, y_pred, cfg, clave):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.scatter(y_real, y_pred, s=10, alpha=0.4, color=COLOR_PUNTOS)
    lim = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
    ax.plot(lim, lim, color=COLOR_RECTA, linewidth=2, label="Predicción perfecta")
    ax.set_xlabel(f"Valor real ({cfg['unidad']})")
    ax.set_ylabel(f"Valor predicho ({cfg['unidad']})")
    ax.set_title("Real vs predicho (conjunto de prueba)")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    residuos = y_real - y_pred
    ax.scatter(y_pred, residuos, s=10, alpha=0.4, color=COLOR_PUNTOS)
    ax.axhline(0, color=COLOR_RECTA, linewidth=2)
    ax.set_xlabel(f"Valor predicho ({cfg['unidad']})")
    ax.set_ylabel("Residuo")
    ax.set_title("Residuos del modelo")
    ax.grid(alpha=0.25)

    fig.suptitle(cfg["nombre"])
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"{clave}_real_vs_predicho.png", dpi=150)
    plt.close(fig)


def graficar_importancia(coeficientes, cfg, clave):
    """Barras con el coeficiente estandarizado (comparable entre variables)."""
    filas = [c for c in coeficientes if c["variable"] != "Intercepto"]
    nombres = [c["variable"] for c in filas]
    betas = [c["beta_estandarizado"] for c in filas]
    colores = [COLOR_PUNTOS if b >= 0 else COLOR_RECTA for b in betas]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(nombres, betas, color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente estandarizado (beta)")
    ax.set_title(f"Importancia relativa - {cfg['nombre']}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(DIR_FIGURAS / f"{clave}_importancia.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Despliegue: exportar el modelo
# --------------------------------------------------------------------------
def exportar_modelo(modelo, cfg, metricas, ruta):
    """
    Guarda en un solo archivo .joblib el modelo y la información que la
    aplicación necesita para usarlo (orden de las variables, unidad, etc.).
    """
    paquete = {
        "modelo": modelo,
        "variables": cfg["variables"],
        "objetivo": cfg["objetivo"],
        "unidad": cfg["unidad"],
        "metricas_prueba": metricas,
    }
    joblib.dump(paquete, ruta)


# --------------------------------------------------------------------------
# Verificaciones adicionales (se citan en el informe)
# --------------------------------------------------------------------------
def verificacion_cronologica_dolar():
    """
    El dólar es una serie de tiempo: además de la división aleatoria, se prueba
    entrenar con el 80 % de los primeros días y evaluar con el 20 % final.
    """
    cfg = EJERCICIOS["dolar"]
    df = cargar_datos(cfg)
    corte = int(len(df) * 0.8)
    X, y = df[cfg["variables"]], df[cfg["objetivo"]]
    modelo = LinearRegression().fit(X[:corte], y[:corte])
    met = evaluar(modelo, X[corte:], y[corte:])
    return {"descripcion": "Entrenamiento con los primeros 400 días, prueba con los últimos 100",
            "metricas": met}


def verificacion_codificacion_energia():
    """
    Hora (1-24) y Dia_Semana (1-7) son variables cíclicas/categóricas. Se compara
    el modelo base contra dos codificaciones alternativas.
    """
    cfg = EJERCICIOS["energia"]
    df = cargar_datos(cfg)
    y = df[cfg["objetivo"]]

    def r2_rmse(X):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TAMANO_PRUEBA, random_state=SEMILLA)
        met = evaluar(LinearRegression().fit(X_tr, y_tr), X_te, y_te)
        return {"R2": met["R2"], "RMSE": met["RMSE"]}

    base = df[cfg["variables"]]

    dias = pd.get_dummies(df["Dia_Semana"], prefix="dia", drop_first=True).astype(float)
    con_dummies = pd.concat([df[["Temperatura", "Hora"]], dias], axis=1)

    ciclica = pd.DataFrame({
        "Temperatura": df["Temperatura"],
        "hora_sen": np.sin(2 * np.pi * df["Hora"] / 24),
        "hora_cos": np.cos(2 * np.pi * df["Hora"] / 24),
        "Dia_Semana": df["Dia_Semana"],
    })

    return {
        "modelo_base": r2_rmse(base),
        "dia_semana_como_categoria": r2_rmse(con_dummies),
        "hora_ciclica_seno_coseno": r2_rmse(ciclica),
        "consumo_medio_por_dia": df.groupby("Dia_Semana")[cfg["objetivo"]].mean().round(2).to_dict(),
        "consumo_medio_por_hora": df.groupby("Hora")[cfg["objetivo"]].mean().round(2).to_dict(),
    }


# --------------------------------------------------------------------------
# Flujo principal
# --------------------------------------------------------------------------
def entrenar_ejercicio(clave, cfg):
    print(f"\n=== {cfg['nombre']} ===")
    df = cargar_datos(cfg)
    resumen = explorar(df)
    print(f"Filas: {resumen['filas']} | Nulos: {resumen['nulos']} | "
          f"Duplicados: {resumen['duplicados']}")

    X = df[cfg["variables"]]
    y = df[cfg["objetivo"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TAMANO_PRUEBA, random_state=SEMILLA)

    modelo = ajustar_modelo(X_train, y_train)

    met_train = evaluar(modelo, X_train, y_train)
    met_test = evaluar(modelo, X_test, y_test)
    coeficientes = analizar_coeficientes(modelo, X_train, y_train, cfg["variables"])

    print(f"R² entrenamiento: {met_train['R2']:.4f} | R² prueba: {met_test['R2']:.4f}")
    print(f"MSE prueba: {met_test['MSE']:.4f} | RMSE prueba: {met_test['RMSE']:.4f}")
    for c in coeficientes:
        print(f"  {c['variable']:<18} coef = {c['coeficiente']:>12.5f}   p = {c['p_valor']:.4g}")

    # gráficas
    graficar_dispersion(df, cfg, clave)
    graficar_correlacion(df, cfg, clave)
    graficar_real_vs_predicho(y_test, modelo.predict(X_test), cfg, clave)
    graficar_importancia(coeficientes, cfg, clave)

    # exportación
    exportar_modelo(modelo, cfg, met_test, DIR_MODELOS / cfg["modelo"])

    return {
        "nombre": cfg["nombre"],
        "unidad": cfg["unidad"],
        "datos": resumen,
        "n_entrenamiento": int(len(X_train)),
        "n_prueba": int(len(X_test)),
        "metricas_entrenamiento": met_train,
        "metricas_prueba": met_test,
        "coeficientes": coeficientes,
        "correlaciones": df.corr().round(4).to_dict(),
    }


def main():
    for carpeta in (DIR_MODELOS, DIR_FIGURAS, DIR_RESULTADOS):
        carpeta.mkdir(exist_ok=True)

    resultados = {clave: entrenar_ejercicio(clave, cfg)
                  for clave, cfg in EJERCICIOS.items()}

    resultados["verificaciones"] = {
        "dolar_cronologico": verificacion_cronologica_dolar(),
        "energia_codificacion": verificacion_codificacion_energia(),
    }
    print("\nVerificación cronológica (dólar):",
          {k: round(v, 4) for k, v in resultados["verificaciones"]["dolar_cronologico"]["metricas"].items()})
    print("Codificación alternativa (energía):",
          {k: v for k, v in resultados["verificaciones"]["energia_codificacion"].items()
           if isinstance(v, dict) and "R2" in v})

    with open(DIR_RESULTADOS / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print("\nListo: modelos en models/, gráficas en figures/, métricas en resultados/resultados.json")


if __name__ == "__main__":
    main()
