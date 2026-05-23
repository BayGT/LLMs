import pandas as pd

def calcular_retencion_semanal(df, user_col, fecha_col):
    data = df.copy()

    data[fecha_col] = pd.to_datetime(data[fecha_col])

    data["semana"] = data[fecha_col].dt.to_period("W-SUN").dt.start_time

    cohortes = data.groupby(user_col)["semana"].min().rename("cohorte")

    data = data.merge(cohortes, on=user_col, how="left")

    data["semana_relativa"] = (
        (data["semana"] - data["cohorte"]).dt.days // 7
    ).astype(int)

    conteos = (
        data.groupby(["cohorte", "semana_relativa"])[user_col]
        .nunique()
        .rename("usuarios_activos")
        .reset_index()
    )

    tam_cohorte = (
        conteos[conteos["semana_relativa"] == 0]
        .set_index("cohorte")["usuarios_activos"]
        .rename("tam_cohorte")
    )

    conteos = conteos.merge(tam_cohorte, on="cohorte", how="left")

    conteos["retencion"] = conteos["usuarios_activos"] / conteos["tam_cohorte"]

    tabla_retencion = conteos.pivot_table(
        index="cohorte",
        columns="semana_relativa",
        values="retencion",
        aggfunc="first",
        fill_value=0.0
    )

    tabla_retencion = tabla_retencion.sort_index().sort_index(axis=1)

    tabla_retencion.columns.name = None

    return tabla_retencion
