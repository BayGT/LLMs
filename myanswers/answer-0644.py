import pandas as pd
from sklearn.impute import KNNImputer

def imputar_con_knn(df, n_neighbors, target_col):
    """
    Imputa los valores faltantes de las columnas numéricas usando KNNImputer,
    dejando la columna objetivo sin modificar y ubicada al final del DataFrame.
    """

    # Separar columnas predictoras y columna objetivo
    features_df = df.drop(columns=[target_col])
    target = df[target_col]

    # Aplicar KNNImputer a las columnas predictoras
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.fit_transform(features_df)

    # Reconstruir el DataFrame con el mismo índice y nombres de columnas
    df_resultado = pd.DataFrame(
        X_imputed,
        columns=features_df.columns,
        index=df.index
    )

    # Agregar la columna objetivo al final
    df_resultado[target_col] = target.values

    return df_resultado
