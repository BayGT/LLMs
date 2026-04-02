import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def generar_caso_de_uso_calcular_importancia_features():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_importancia_features.
    """
    
    # 1. Configuración aleatoria
    n_samples = random.randint(30, 80)
    n_features = random.randint(3, 6)
    
    columnas = [f'feature_{i}' for i in range(n_features)]
    
    # 2. Generar datos aleatorios
    X = np.random.randn(n_samples, n_features)
    
    # Generar relación con pesos ocultos
    pesos_reales = np.random.uniform(-2, 2, size=n_features)
    logits = X @ pesos_reales
    probs = 1 / (1 + np.exp(-logits))
    
    y = (probs > 0.5).astype(int)
    
    df = pd.DataFrame(X, columns=columnas)
    columna_objetivo = 'target'
    df[columna_objetivo] = y
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'columnas': columnas,
        'columna_objetivo': columna_objetivo
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    
    X_expected = df[columnas].to_numpy()
    y_expected = df[columna_objetivo].to_numpy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_expected)
    
    model = LogisticRegression()
    model.fit(X_scaled, y_expected)
    
    importancia = np.abs(model.coef_[0])
    
    output_data = importancia
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_calcular_importancia_features()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Columnas predictoras: {entrada['columnas']}")
    print(f"Columna objetivo: {entrada['columna_objetivo']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (Array de numpy) ===")
    print(f"Shape del array de importancia: {salida_esperada.shape}")
    print("Importancia de features:")
    print(salida_esperada)
