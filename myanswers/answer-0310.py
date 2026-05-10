entrada, salida_esperada = generar_caso_de_uso_evaluar_lift_por_deciles()
def evaluar_lift_por_deciles(df, score_col, target_col, n_bins=5):
    """
    Solución del caso de uso:
    calcula una tabla de lift por grupos de score y el ROC-AUC.
    """

    # 1. Copiar el DataFrame para no modificar el original
    data = df.copy()

    # 2. Calcular la tasa global de positivos
    tasa_global = data[target_col].mean()

    # 3. Crear los segmentos con pd.qcut
    segmentos = pd.qcut(
        data[score_col],
        q=n_bins,
        duplicates="drop"
    )

    # 4. Calcular métricas por segmento
    tabla_lift = (
        data.assign(segmento_intervalo=segmentos)
        .groupby("segmento_intervalo", observed=False)
        .agg(
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
            n=(target_col, "size"),
            tasa_positivos=(target_col, "mean")
        )
        .reset_index()
    )

    # 5. Calcular el lift
    tabla_lift["lift"] = tabla_lift["tasa_positivos"] / tasa_global

    # 6. Ordenar desde los scores más altos hasta los más bajos
    tabla_lift = (
        tabla_lift
        .sort_values("score_max", ascending=False)
        .reset_index(drop=True)
    )

    # 7. Crear la columna segmento
    tabla_lift["segmento"] = np.arange(1, len(tabla_lift) + 1)

    # 8. Dejar las columnas solicitadas y en el orden correcto
    tabla_lift = tabla_lift[
        ["segmento", "score_min", "score_max", "n", "tasa_positivos", "lift"]
    ].copy()

    # 9. Calcular el ROC-AUC
    auc = float(roc_auc_score(data[target_col], data[score_col]))

    # 10. Devolver el diccionario solicitado
    return {
        "tabla_lift": tabla_lift,
        "auc": auc,
        "n_segmentos": int(len(tabla_lift))
    }


resultado = evaluar_lift_por_deciles(**entrada)

print("=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")

print("\nTabla lift:")
print(resultado["tabla_lift"])

print("\nAUC:")
print(resultado["auc"])

print("\nNúmero de segmentos:")
print(resultado["n_segmentos"])

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")

print("\nTabla lift esperada:")
print(salida_esperada["tabla_lift"])

print("\nAUC esperado:")
print(salida_esperada["auc"])

print("\nNúmero de segmentos esperado:")
print(salida_esperada["n_segmentos"])

print("\n=== COMPROBACIÓN ===")

try:
    pd.testing.assert_frame_equal(
        resultado["tabla_lift"],
        salida_esperada["tabla_lift"],
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10
    )

    auc_correcto = np.isclose(resultado["auc"], salida_esperada["auc"])
    segmentos_correctos = resultado["n_segmentos"] == salida_esperada["n_segmentos"]

    print(auc_correcto and segmentos_correctos)

except AssertionError as error:
    print(False)
    print(error)
