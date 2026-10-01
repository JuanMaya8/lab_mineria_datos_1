"""
Ejemplo mínimo de cómo cargar un modelo exportado (.joblib) y usarlo
sin necesidad de la interfaz web.

Uso:
    python cargar_modelo_ejemplo.py
"""
import joblib
import pandas as pd

from config import DIR_MODELOS

# cada .joblib guarda un diccionario con el modelo y datos de apoyo
paquete = joblib.load(DIR_MODELOS / "modelo_glucosa.joblib")
modelo = paquete["modelo"]

# los valores deben ir en el mismo orden con el que se entrenó el modelo
paciente = pd.DataFrame([{"Edad": 55, "IMC": 28.0, "Actividad_Fisica": 3}],
                        columns=paquete["variables"])

prediccion = modelo.predict(paciente)[0]
print(f"{paquete['objetivo']} estimado: {prediccion:.2f} {paquete['unidad']}")
