import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generar_caso_de_uso_evaluar_overfitting():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función evaluar_overfitting.
    """
    
    # 1. Configuración aleatoria
    n_samples = random.randint(40, 100)
    n_features = random.randint(2, 5)
    
    columnas = [f'feature_{i}' for i in range(n_features)]
    
    # 2. Generar datos con relación lineal + ruido
    X = np.random.randn(n_samples, n_features)
    coef = np.random.uniform(-3, 3, size=n_features)
    y = X @ coef + np.random.normal(0, 2, size=n_samples)
    
    df = pd.DataFrame(X, columns=columnas)
    columna_objetivo = 'target'
    df[columna_objetivo] = y
    
    test_size = random.choice([0.2, 0.25, 0.3])
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'columnas': columnas,
        'columna_objetivo': columna_objetivo,
        'test_size': test_size
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    
    X_data = df[columnas].to_numpy()
    y_data = df[columna_objetivo].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    output_data = np.array([mse_train, mse_test])
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_evaluar_overfitting()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Columnas predictoras: {entrada['columnas']}")
    print(f"Columna objetivo: {entrada['columna_objetivo']}")
    print(f"Test size: {entrada['test_size']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (Array de numpy) ===")
    print(f"Shape del resultado: {salida_esperada.shape}")
    print("MSE entrenamiento y test:")
    print(salida_esperada)
