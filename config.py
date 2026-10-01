"""
Configuración central del laboratorio.

Aquí se define, para cada ejercicio, qué archivo CSV usar, cuáles son las
variables independientes (X), cuál es la variable dependiente (y) y los
rangos que usa la interfaz web. Si se quiere agregar un cuarto ejercicio,
basta con añadir una entrada nueva al diccionario EJERCICIOS.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DIR_DATOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_FIGURAS = RAIZ / "figures"
DIR_RESULTADOS = RAIZ / "resultados"

SEMILLA = 42          # para que los resultados sean reproducibles
TAMANO_PRUEBA = 0.20  # 80 % entrenamiento / 20 % prueba

EJERCICIOS = {
    "dolar": {
        "nombre": "Precio del dólar",
        "csv": "dolar_data.csv",
        "modelo": "modelo_dolar.joblib",
        "variables": ["Dia", "Inflacion", "Tasa_interes"],
        "objetivo": "Precio_Dolar",
        "unidad": "COP",
        "metrica_principal": "MSE",
        # (mínimo, máximo, valor por defecto, paso) para cada entrada de la interfaz
        "rangos": {
            "Dia": (1, 1000, 250, 1),
            "Inflacion": (0.0, 0.10, 0.02, 0.001),
            "Tasa_interes": (0.0, 15.0, 5.0, 0.1),
        },
        "etiquetas": {
            "Dia": "Día (número consecutivo)",
            "Inflacion": "Inflación diaria",
            "Tasa_interes": "Tasa de interés diaria",
        },
    },
    "glucosa": {
        "nombre": "Nivel de glucosa en sangre",
        "csv": "glucosa_data.csv",
        "modelo": "modelo_glucosa.joblib",
        "variables": ["Edad", "IMC", "Actividad_Fisica"],
        "objetivo": "Nivel_Glucosa",
        "unidad": "mg/dL",
        "metrica_principal": "MSE",
        "rangos": {
            "Edad": (18, 100, 45, 1),
            "IMC": (12.0, 50.0, 25.0, 0.1),
            "Actividad_Fisica": (0, 20, 4, 1),
        },
        "etiquetas": {
            "Edad": "Edad (años)",
            "IMC": "Índice de masa corporal (IMC)",
            "Actividad_Fisica": "Actividad física (horas por semana)",
        },
    },
    "energia": {
        "nombre": "Consumo de energía eléctrica",
        "csv": "energia_data.csv",
        "modelo": "modelo_energia.joblib",
        "variables": ["Temperatura", "Hora", "Dia_Semana"],
        "objetivo": "Consumo_Energia",
        "unidad": "kWh",
        "metrica_principal": "RMSE",
        "rangos": {
            "Temperatura": (-5.0, 50.0, 25.0, 0.1),
            "Hora": (1, 24, 12, 1),
            "Dia_Semana": (1, 7, 3, 1),
        },
        "etiquetas": {
            "Temperatura": "Temperatura (°C)",
            "Hora": "Hora del día (1 a 24)",
            "Dia_Semana": "Día de la semana (1=Lunes ... 7=Domingo)",
        },
    },
}
