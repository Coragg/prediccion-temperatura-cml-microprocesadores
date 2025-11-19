import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

def load_and_preprocess_data(filepath):
    """Carga y preprocesa los datos de temperatura"""
    df = pd.read_csv(filepath, index_col=0)

    # Convertir fecha a features numericos
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.sort_values('Date').reset_index(drop=True)

    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

    # Features ciclicas para capturar estacionalidad
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Codificar variables categoricas
    le_weather = LabelEncoder()
    le_cloud = LabelEncoder()

    df['weather_encoded'] = le_weather.fit_transform(df['weather'].fillna('Unknown'))
    df['cloud_encoded'] = le_cloud.fit_transform(df['cloud'].fillna('Unknown'))

    # Features de lag temporal (dias anteriores)
    for lag in [1, 2, 3, 7]:
        df[f'maxtemp_lag{lag}'] = df['maxtemp'].shift(lag)
        df[f'mintemp_lag{lag}'] = df['mintemp'].shift(lag)

    # Promedios moviles
    df['maxtemp_rolling3'] = df['maxtemp'].rolling(window=3).mean().shift(1)
    df['maxtemp_rolling7'] = df['maxtemp'].rolling(window=7).mean().shift(1)
    df['mintemp_rolling3'] = df['mintemp'].rolling(window=3).mean().shift(1)

    # Diferencia de temperatura del dia anterior
    df['temp_diff_lag1'] = df['maxtemp'].shift(1) - df['mintemp'].shift(1)

    # Features adicionales
    df['temp_range_proxy'] = df['pressure'] * df['humidity'] / 1000
    df['wind_humidity'] = df['mean wind speed'] * df['humidity'] / 100
    df['pressure_diff'] = df['pressure'].diff().shift(1)

    # Eliminar filas con NaN por los lags
    df = df.dropna()

    return df

def prepare_features(df):
    """Prepara las features para el modelo"""
    feature_columns = [
        'mintemp', 'pressure', 'humidity', 'mean wind speed',
        'month', 'day', 'day_of_year', 'week_of_year',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'weather_encoded', 'cloud_encoded',
        'temp_range_proxy', 'wind_humidity', 'pressure_diff',
        # Features de lag temporal
        'maxtemp_lag1', 'maxtemp_lag2', 'maxtemp_lag3', 'maxtemp_lag7',
        'mintemp_lag1', 'mintemp_lag2', 'mintemp_lag3', 'mintemp_lag7',
        # Promedios moviles
        'maxtemp_rolling3', 'maxtemp_rolling7', 'mintemp_rolling3',
        'temp_diff_lag1'
    ]

    X = df[feature_columns]
    y = df['maxtemp']  # Predecir temperatura maxima

    return X, y

def train_model(X_train, y_train):
    """Entrena el modelo Random Forest optimizado"""
    model = RandomForestRegressor(
        n_estimators=500,        # Mas arboles
        max_depth=None,          # Sin limite de profundidad
        min_samples_split=2,     # Permitir splits finos
        min_samples_leaf=1,      # Minimo en hojas
        max_features=0.8,        # 80% de features por arbol
        bootstrap=True,
        oob_score=True,          # Validacion out-of-bag
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evalua el modelo y retorna metricas"""
    y_pred = model.predict(X_test)

    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    return metrics, y_pred

def main():
    # Cargar datos
    df = load_and_preprocess_data('data/data.csv')

    # Preparar features
    X, y = prepare_features(df)

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo
    print("Entrenando modelo Random Forest optimizado...")
    model = train_model(X_train, y_train)

    # Evaluar modelo
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Mostrar metricas
    print("\n## Metricas del Modelo Random Forest")
    print(f"- **MSE**: {metrics['mse']:.4f}")
    print(f"- **RMSE**: {metrics['rmse']:.4f}")
    print(f"- **MAE**: {metrics['mae']:.4f}")
    print(f"- **R2**: {metrics['r2']:.4f}")

    # Guardar metricas en JSON para CML
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Guardar predicciones para visualizacion
    results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred
    })
    results.to_csv('predictions.csv', index=False)

    # Guardar importancia de features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('feature_importance.csv', index=False)

    print("\nArchivos generados: metrics.json, predictions.csv, feature_importance.csv")

    return model, metrics

if __name__ == "__main__":
    main()
