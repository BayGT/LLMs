def imputar_con_knn(df, n_neighbors, target_col):
    """
    Solución del caso de uso:
    imputa los valores faltantes de las columnas numéricas usando KNNImputer,
    dejando la columna objetivo sin modificar y al final del DataFrame.
    """

    # 1. Copiar el DataFrame para no modificar el original
    data = df.copy()

    # 2. Separar las columnas predictoras de la columna objetivo
    features_df = data.drop(columns=[target_col])
    target = data[target_col]

    # 3. Aplicar KNNImputer a las columnas predictoras
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.fit_transform(features_df)

    # 4. Construir un nuevo DataFrame con las columnas imputadas
    df_resultado = pd.DataFrame(
        X_imputed,
        columns=features_df.columns,
        index=data.index
    )

    # 5. Agregar la columna objetivo al final con sus valores originales
    df_resultado[target_col] = target.values

    # 6. Devolver el DataFrame final
    return df_resultado


# Comprobación de la función solución usando el caso de uso generado

resultado = imputar_con_knn(**entrada)

print("=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print(resultado.head(6).to_string())

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")
print(salida_esperada.head(6).to_string())

print("\n=== COMPROBACIÓN ===")

try:
    pd.testing.assert_frame_equal(resultado, salida_esperada)
    print(True)
except AssertionError:
    print(False)
