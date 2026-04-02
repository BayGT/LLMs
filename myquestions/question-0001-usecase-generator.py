import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def generar_caso_de_uso_evaluar_calidad_clustering():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función evaluar_calidad_clustering.
    """
    
    # 1. Configuración aleatoria
    n_clusters = random.randint(2, 4)
    puntos_por_cluster = random.randint(5, 10)
    n_features = random.randint(2, 4)
    
    columnas = [f'feature_{i}' for i in range(n_features)]
    
    # 2. Generar datos artificiales con clusters relativamente separados
    data = []
    
    for _ in range(n_clusters):
        centro = np.random.uniform(-10, 10, size=n_features)
        puntos = centro + np.random.normal(0, 1.0, size=(puntos_por_cluster, n_features))
        data.append(puntos)
    
    data = np.vstack(data)
    np.random.shuffle(data)
    
    df = pd.DataFrame(data, columns=columnas)
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'columnas': columnas,
        'n_clusters': n_clusters
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    #    Aquí replicamos la lógica que debería tener
    #    la función evaluar_calidad_clustering
    # ---------------------------------------------------------
    
    X_expected = df[columnas].to_numpy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_expected)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, labels)
    
    output_data = score
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_evaluar_calidad_clustering()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Columnas seleccionadas: {entrada['columnas']}")
    print(f"Número de clusters: {entrada['n_clusters']}")
    print("DataFrame (primeras 5 filas):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Silhouette Score esperado: {salida_esperada}")
