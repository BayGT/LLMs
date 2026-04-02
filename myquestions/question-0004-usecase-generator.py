import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def generar_caso_de_uso_evaluar_overfitting():
    # ---------------------------
    # 1. Generar datos aleatorios
    # ---------------------------
    n_samples = random.randint(40, 100)
    n_features = random.randint(2, 5)

    columnas = [f"feature_{i}" for i in range(n_features)]

    X = np.random.randn(n_samples, n_features)

    # Generar relación con ruido
    coef = np.random.uniform(-3, 3, size=n_features)
    y = X @ coef + np.random.normal(0, 2, size=n_samples)

    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y

    test_size = random.choice([0.2, 0.25, 0.3])

    # ---------------------------
    # 2. Construir input
    # ---------------------------
    input_data = {
        "df": df.copy(),
        "columnas": columnas,
        "columna_objetivo": "target",
        "test_size": test_size
    }

    # ---------------------------
    # 3. Calcular output esperado
    # ---------------------------
    X_data = df[columnas].values
    y_data = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    output_data = np.array([mse_train, mse_test])

    return input_data, output_data


# Ejemplo de uso
generar_caso_de_uso_evaluar_overfitting()
