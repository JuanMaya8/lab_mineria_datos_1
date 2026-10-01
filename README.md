# Laboratorio 1 · Minería de Datos — Regresión lineal múltiple (CRISP-DM)

Tres modelos de regresión lineal múltiple (precio del dólar, nivel de glucosa y consumo de energía), exportados con `joblib` y usados desde una interfaz web hecha con Streamlit.

## 1. Cómo ejecutarlo

Requiere Python 3.9 o superior. Desde la carpeta del proyecto:

```bash
# (opcional pero recomendado) entorno virtual
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

**Paso 1 – Entrenar los modelos** (genera los `.joblib`, las gráficas y las métricas):

```bash
python entrenamiento.py
```

**Paso 2 – Abrir la interfaz web:**

```bash
streamlit run app.py
```

Se abre en el navegador (normalmente `http://localhost:8501`). En el menú lateral se elige el ejercicio, se escriben los valores y se pulsa **Predecir**.

> El zip ya incluye los modelos y las gráficas generadas, así que se puede ir directo al Paso 2. El Paso 1 solo hace falta si se quiere volver a entrenar.

**Opcional – usar un modelo sin interfaz:**

```bash
python cargar_modelo_ejemplo.py
```

## 2. Estructura del proyecto

```
laboratorio_mineria_datos/
├── app.py                      # Interfaz web (Streamlit)
├── entrenamiento.py            # Pipeline CRISP-DM: entrena, evalúa, grafica y exporta
├── config.py                   # Variables, rutas y rangos de cada ejercicio
├── cargar_modelo_ejemplo.py    # Ejemplo de cómo cargar un .joblib
├── requirements.txt
├── data/                       # CSV originales (dolar, glucosa, energia)
├── models/                     # Modelos exportados (.joblib)
├── figures/                    # Gráficas en PNG
└── resultados/resultados.json  # Métricas y coeficientes de los tres modelos
```

## 3. Dónde está cada parte importante

| Qué se busca | Archivo | Función / bloque |
|---|---|---|
| Qué variables usa cada ejercicio (X e y) | `config.py` | diccionario `EJERCICIOS` |
| Semilla y proporción entrenamiento/prueba | `config.py` | `SEMILLA`, `TAMANO_PRUEBA` |
| Carga y validación de los CSV | `entrenamiento.py` | `cargar_datos()` |
| Calidad de datos (nulos, duplicados, estadísticas) | `entrenamiento.py` | `explorar()` |
| **Ajuste del modelo de regresión lineal** | `entrenamiento.py` | `ajustar_modelo()` |
| **MSE, RMSE, MAE y R²** | `entrenamiento.py` | `evaluar()` |
| **Coeficientes, p-valores y coeficientes estandarizados** (importancia de variables) | `entrenamiento.py` | `analizar_coeficientes()` |
| Gráficas de dispersión de cada variable vs. la dependiente | `entrenamiento.py` | `graficar_dispersion()` |
| Matriz de correlación | `entrenamiento.py` | `graficar_correlacion()` |
| Real vs. predicho y residuos | `entrenamiento.py` | `graficar_real_vs_predicho()` |
| Gráfica de importancia de variables | `entrenamiento.py` | `graficar_importancia()` |
| **Exportación del modelo** (`joblib.dump`) | `entrenamiento.py` | `exportar_modelo()` |
| Orden general de los pasos de cada ejercicio | `entrenamiento.py` | `entrenar_ejercicio()` y `main()` |
| **Carga del modelo exportado** (`joblib.load`) | `app.py` | `cargar_modelo()` |
| Campos de entrada por teclado | `app.py` | `crear_entrada()` |
| Selector de ejercicio (Dólar / Glucosa / Energía) | `app.py` | `st.radio` dentro de `with st.sidebar` |
| **Cálculo y visualización de la predicción** | `app.py` | botón "Predecir" en `with tab_pred` |
| Aviso de extrapolación (valores fuera del rango de entrenamiento) | `app.py` | `rangos_entrenamiento()` y bloque `fuera` |

### Fases de CRISP-DM en el código

1. **Comprensión de los datos** → `explorar()`, `graficar_correlacion()`, `graficar_dispersion()`
2. **Preparación de los datos** → `cargar_datos()` y `train_test_split` en `entrenar_ejercicio()`
3. **Modelado** → `ajustar_modelo()`
4. **Evaluación** → `evaluar()`, `analizar_coeficientes()`, `graficar_real_vs_predicho()`
5. **Despliegue** → `exportar_modelo()` y `app.py`

## 4. Qué contiene cada archivo `.joblib`

Cada archivo de `models/` es un diccionario con:

- `modelo`: el `LinearRegression` entrenado
- `variables`: nombres de las variables en el orden en que se entrenó
- `objetivo` y `unidad`: variable dependiente y su unidad
- `metricas_prueba`: MSE, RMSE, MAE y R² sobre el conjunto de prueba

## 5. Resultados (conjunto de prueba, 80 % / 20 %, semilla 42)

| Ejercicio | MSE | RMSE | R² |
|---|---|---|---|
| Dólar | 2 376,97 | 48,75 | 0,9963 |
| Glucosa | 233,69 | 15,29 | 0,6814 |
| Energía | 429,52 | 20,72 | 0,8968 |

El análisis completo está en el informe de Word.

## 6. Notas

- Si al cargar un `.joblib` aparece una advertencia de versión de scikit-learn, vuelva a ejecutar `python entrenamiento.py` para regenerarlos con la versión instalada.
- Para agregar un ejercicio nuevo basta con añadir una entrada en `EJERCICIOS` (`config.py`) y un CSV en `data/`; el entrenamiento y la interfaz lo recogen sin más cambios.
