# Prediccion de Temperatura con Random Forest

Pipeline de Machine Learning con CML (Continuous Machine Learning) para predecir temperatura utilizando Random Forest.

## Descripcion

Este proyecto implementa un modelo de prediccion de temperatura maxima basado en variables meteorologicas como presion, humedad, velocidad del viento y condiciones climaticas.

## Estructura del Proyecto

```
.
├── .github/
│   └── workflows/
│       └── cml.yaml          # Pipeline CI/CD con CML
├── data/
│   └── data.csv              # Dataset meteorologico
├── src/
│   ├── train.py              # Entrenamiento del modelo
│   └── cluster_plot.py       # Generacion de visualizaciones
├── pyproject.toml            # Dependencias del proyecto
└── README.md
```

## Modelo

**Algoritmo**: Random Forest Regressor

**Features utilizadas**:
- Temperatura minima
- Presion atmosferica
- Humedad
- Velocidad media del viento
- Mes, dia y dia del año
- Condiciones climaticas (codificadas)
- Nubosidad (codificada)

**Variable objetivo**: Temperatura maxima

## Pipeline CML

El workflow de GitHub Actions ejecuta automaticamente:

1. Entrenamiento del modelo Random Forest
2. Evaluacion con metricas (MSE, RMSE, MAE, R²)
3. Generacion de visualizaciones
4. Publicacion del reporte como comentario en el commit

### Metricas Generadas

- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coeficiente de determinacion)

### Visualizaciones

El script `cluster_plot.py` genera:
- Clusters de temperatura maxima vs minima
- Clusters de presion vs humedad
- Grafico de prediccion vs valores reales
- Importancia de features

## Instalacion Local

```bash
# Clonar repositorio
git clone https://github.com/Coragg/prediccion-temperatura-cml-microprocesadores.git
cd prediccion-temperatura-cml-microprocesadores

# Instalar dependencias
pip install numpy pandas scipy scikit-learn matplotlib

# Ejecutar entrenamiento
python src/train.py

# Generar visualizaciones
python src/cluster_plot.py
```

## Archivos Generados

Despues de ejecutar el pipeline:

| Archivo | Descripcion |
|---------|-------------|
| `metrics.json` | Metricas del modelo en formato JSON |
| `predictions.csv` | Predicciones vs valores reales |
| `feature_importance.csv` | Importancia de cada feature |
| `cluster_plot.png` | Visualizaciones del analisis |

## Dependencias

- Python 3.11+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SciPy

## Autores

- Victor Camero

---

*Pipeline automatizado con [CML](https://cml.dev/) en GitHub Actions*
