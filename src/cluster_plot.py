import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def create_cluster_plot():
    """Crea visualización de clusters basada en datos meteorológicos"""

    # Cargar datos
    df = pd.read_csv('data/data.csv', index_col=0)

    # Seleccionar features numéricas para clustering
    features = ['maxtemp', 'mintemp', 'pressure', 'humidity', 'mean wind speed']
    X = df[features].dropna()

    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Crear figura con múltiples subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Temperatura máxima vs mínima con clusters
    scatter1 = axes[0, 0].scatter(
        X['maxtemp'], X['mintemp'],
        c=clusters, cmap='viridis', alpha=0.6
    )
    axes[0, 0].set_xlabel('Temperatura Máxima (°C)')
    axes[0, 0].set_ylabel('Temperatura Mínima (°C)')
    axes[0, 0].set_title('Clusters: Temp Máxima vs Mínima')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')

    # Plot 2: Presión vs Humedad con clusters
    scatter2 = axes[0, 1].scatter(
        X['pressure'], X['humidity'],
        c=clusters, cmap='viridis', alpha=0.6
    )
    axes[0, 1].set_xlabel('Presión (hPa)')
    axes[0, 1].set_ylabel('Humedad (%)')
    axes[0, 1].set_title('Clusters: Presión vs Humedad')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')

    # Plot 3: Predicciones vs Valores Reales (si existe el archivo)
    try:
        predictions = pd.read_csv('predictions.csv')
        axes[1, 0].scatter(
            predictions['actual'], predictions['predicted'],
            alpha=0.6, color='blue'
        )
        # Línea de referencia perfecta
        min_val = min(predictions['actual'].min(), predictions['predicted'].min())
        max_val = max(predictions['actual'].max(), predictions['predicted'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 0].set_xlabel('Temperatura Real (°C)')
        axes[1, 0].set_ylabel('Temperatura Predicha (°C)')
        axes[1, 0].set_title('Random Forest: Predicción vs Real')
    except FileNotFoundError:
        axes[1, 0].text(0.5, 0.5, 'Ejecutar train.py primero',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Predicciones (No disponible)')

    # Plot 4: Importancia de Features (si existe el archivo)
    try:
        importance = pd.read_csv('feature_importance.csv')
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance)))
        bars = axes[1, 1].barh(importance['feature'], importance['importance'], color=colors)
        axes[1, 1].set_xlabel('Importancia')
        axes[1, 1].set_title('Importancia de Features - Random Forest')
    except FileNotFoundError:
        axes[1, 1].text(0.5, 0.5, 'Ejecutar train.py primero',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance (No disponible)')

    plt.tight_layout()
    plt.savefig('cluster_plot.png', dpi=150, bbox_inches='tight')
    print("Gráfico guardado: cluster_plot.png")

    return fig

if __name__ == "__main__":
    create_cluster_plot()
