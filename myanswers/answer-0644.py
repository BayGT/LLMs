import pandas as pd
from sklearn.impute import KNNImputer

def imputar_con_knn(df, n_neighbors, target_col):
    features_df = df.drop(columns=[target_col])
    target = df[target_col]

    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.fit_transform(features_df)

    df_resultado = pd.DataFrame(
        X_imputed,
        columns=features_df.columns,
        index=df.index
    )

    df_resultado[target_col] = target.values

    return df_resultado
