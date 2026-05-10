def calcular_retencion_semanal(df, user_col, fecha_col):
    """
    Solución del caso de uso:
    calcula una matriz de retención semanal por cohortes.
    """

    # 1. Copiar el DataFrame para no modificar el original
    data = df.copy()

    # 2. Convertir la columna de fecha a datetime
    data[fecha_col] = pd.to_datetime(data[fecha_col])

    # 3. Asignar cada actividad a la semana calendario que inicia en lunes
    data["semana"] = data[fecha_col].dt.to_period("W-SUN").dt.start_time

    # 4. Determinar la cohorte de cada usuario:
    # la primera semana en la que aparece activo
    cohortes = data.groupby(user_col)["semana"].min().rename("cohorte")

    # 5. Agregar la cohorte al DataFrame original
    data = data.merge(cohortes, on=user_col, how="left")

    # 6. Calcular la semana relativa respecto a la cohorte
    data["semana_relativa"] = (
        (data["semana"] - data["cohorte"]).dt.days // 7
    ).astype(int)

    # 7. Contar usuarios únicos activos por cohorte y semana relativa
    conteos = (
        data.groupby(["cohorte", "semana_relativa"])[user_col]
        .nunique()
        .rename("usuarios_activos")
        .reset_index()
    )

    # 8. Calcular el tamaño de cada cohorte
    tam_cohorte = (
        conteos[conteos["semana_relativa"] == 0]
        .set_index("cohorte")["usuarios_activos"]
        .rename("tam_cohorte")
    )

    # 9. Unir el tamaño de cohorte con los conteos
    conteos = conteos.merge(tam_cohorte, on="cohorte", how="left")

    # 10. Calcular la retención
    conteos["retencion"] = conteos["usuarios_activos"] / conteos["tam_cohorte"]

    # 11. Construir la matriz de retención
    tabla_retencion = conteos.pivot_table(
        index="cohorte",
        columns="semana_relativa",
        values="retencion",
        aggfunc="first",
        fill_value=0.0
    )

    # 12. Ordenar filas y columnas
    tabla_retencion = tabla_retencion.sort_index().sort_index(axis=1)

    # 13. Quitar el nombre de las columnas para que coincida con el esperado
    tabla_retencion.columns.name = None

    return tabla_retencion


# Comprobación de la función solución usando el caso de uso generado

resultado = calcular_retencion_semanal(**entrada)

print("=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print(resultado)

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")
print(salida)

print("\n=== COMPROBACIÓN ===")

try:
    pd.testing.assert_frame_equal(resultado, salida)
    print(True)
except AssertionError:
    print(False)
