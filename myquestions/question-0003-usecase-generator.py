import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

def generar_caso_de_uso_calcular_vif():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_vif.
    """
    
    # 1. Configuración aleatoria
    n_samples = random.randint(30, 80)
    n_features = random.randint(3, 6)
    
    columnas = [f'feature_{i}' for i in range(n_features)]
    
    # 2. Generar datos base
    base = np.random.randn(n_samples, n_features)
    
    # Introducir multicolinealidad artificial
    if n_features > 2:
        base[:, 1] = base[:, 0] * np.random.uniform(0.7, 0.95) + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame(base, columns=columnas)
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'columnas': columnas
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    
    vif_values = []
    
    for col in columnas:
        y = df[col].to_numpy()
        X = df[[c for c in columnas if c != col]].to_numpy()
        
        model = LinearRegression()
        model.fit(X, y)
        
        r2 = model.score(X, y)
        
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vif_values.append(vif)
    
    output_data = np.array(vif_values)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_calcular_vif()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Columnas analizadas: {entrada['columnas']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (Array de numpy) ===")
    print(f"Shape del array VIF: {salida_esperada.shape}")
    print("Valores VIF:")
    print(salida_esperada)
