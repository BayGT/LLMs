import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def generar_caso_de_uso_importancia_features():
    # ---------------------------
    # 1. Generar datos aleatorios
    # ---------------------------
    n_samples = random.randint(30, 80)
    n_features = random.randint(3, 6)

    columnas = [f"feature_{i}" for i in range(n_features)]

    X = np.random.randn(n_samples, n_features)

    # Generar una relación real con pesos ocultos
    pesos_reales = np.random.uniform(-2, 2, size=n_features)
    logits = X @ pesos_reales
    probs = 1 / (1 + np.exp(-logits))

    y = (probs > 0.5).astype(int)

    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y

    # ---------------------------
    # 2. Construir input
    # ---------------------------
    input_data = {
        "df": df.copy(),
        "columnas": columnas,
        "columna_objetivo": "target"
    }

    # ---------------------------
    # 3. Simular resultado esperado
    # ---------------------------
    X_data = df[columnas].values
    y_data = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    model = LogisticRegression()
    model.fit(X_scaled, y_data)

    importancia = np.abs(model.coef_[0])

    output_data = importancia

    return input_data, output_data


# Ejemplo de uso
generar_caso_de_uso_importancia_features()
