import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression


def generar_caso_de_uso_calcular_vif():
    # ---------------------------
    # 1. Generar datos aleatorios
    # ---------------------------
    n_samples = random.randint(30, 80)
    n_features = random.randint(3, 6)

    columnas = [f"feature_{i}" for i in range(n_features)]

    # Generar datos base
    base = np.random.randn(n_samples, n_features)

    # Introducir correlación (multicolinealidad)
    if n_features > 2:
        base[:, 1] = base[:, 0] * np.random.uniform(0.7, 0.95) + np.random.normal(0, 0.1, n_samples)

    df = pd.DataFrame(base, columns=columnas)

    # ---------------------------
    # 2. Construir input
    # ---------------------------
    input_data = {
        "df": df.copy(),
        "columnas": columnas
    }

    # ---------------------------
    # 3. Calcular output esperado
    # ---------------------------
    vif_values = []

    for i, col in enumerate(columnas):
        y = df[col].values
        X = df[[c for c in columnas if c != col]].values

        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)

        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vif_values.append(vif)

    output_data = np.array(vif_values)

    return input_data, output_data


# Ejemplo de uso
generar_caso_de_uso_calcular_vif()
