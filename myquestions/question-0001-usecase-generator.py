import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def generar_caso_de_uso_evaluar_clustering():
    # ---------------------------
    # 1. Generar datos aleatorios
    # ---------------------------
    n_samples = random.randint(20, 60)
    n_features = random.randint(2, 5)
    n_clusters = random.randint(2, 4)

    columnas = [f"feature_{i}" for i in range(n_features)]

    # Crear clusters artificiales
    data = []
    for _ in range(n_clusters):
        centro = np.random.uniform(-10, 10, size=n_features)
        puntos = centro + np.random.normal(0, 1, size=(n_samples // n_clusters, n_features))
        data.append(puntos)

    data = np.vstack(data)

    df = pd.DataFrame(data, columns=columnas)

    # ---------------------------
    # 2. Construir input
    # ---------------------------
    input_data = {
        "df": df.copy(),
        "columnas": columnas,
        "n_clusters": n_clusters
    }

    # ---------------------------
    # 3. Simular resultado esperado
    # ---------------------------
    X = df[columnas].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    output_data = score

    return input_data, output_data


# Ejemplo de uso
generar_caso_de_uso_evaluar_clustering()
