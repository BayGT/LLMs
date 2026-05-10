def evaluar_lift_por_deciles(df, score_col, target_col, n_bins=5):
    """
    Solución del caso de uso:
    evalúa una tabla de lift por segmentos de score y calcula el ROC-AUC.
    """

    # 1. Copiar el DataFrame para no modificar el original
    data = df.copy()

    # 2. Calcular la tasa global de positivos
    tasa_global = data[target_col].mean()

    # 3. Dividir los registros en grupos de igual frecuencia usando qcut
    segmentos = pd.qcut(
        data[score_col],
        q=n_bins,
        duplicates="drop"
    )

    # 4. Construir la tabla de lift por segmento
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
    tabla_lift = tabla_lift.sort_values("score_max", ascending=False).reset_index(drop=True)

    # 7. Crear el número de segmento
    tabla_lift["segmento"] = np.arange(1, len(tabla_lift) + 1)

    # 8. Dejar solamente las columnas solicitadas y en el orden pedido
    tabla_lift = tabla_lift[
        ["segmento", "score_min", "score_max", "n", "tasa_positivos", "lift"]
    ].copy()

    # 9. Calcular el ROC-AUC
    auc = float(roc_auc_score(data[target_col], data[score_col]))

    # 10. Construir el diccionario de salida
    resultado = {
        "tabla_lift": tabla_lift,
        "auc": auc,
        "n_segmentos": int(len(tabla_lift))
    }

    return resultado


# Comprobación de la función solución usando el caso de uso generado

entrada, salida_esperada = generar_caso_de_uso_evaluar_lift_por_deciles()

resultado = evaluar_lift_por_deciles(**entrada)

print("=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print("Tabla lift:")
print(resultado["tabla_lift"])
print("\nAUC:")
print(resultado["auc"])
print("\nNúmero de segmentos:")
print(resultado["n_segmentos"])

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")
print("Tabla lift esperada:")
print(salida_esperada["tabla_lift"])
print("\nAUC esperado:")
print(salida_esperada["auc"])
print("\nNúmero de segmentos esperado:")
print(salida_esperada["n_segmentos"])

print("\n=== COMPROBACIÓN ===")

try:
    pd.testing.assert_frame_equal(resultado["tabla_lift"], salida_esperada["tabla_lift"])
    auc_correcto = np.isclose(resultado["auc"], salida_esperada["auc"])
    segmentos_correctos = resultado["n_segmentos"] == salida_esperada["n_segmentos"]

    print(auc_correcto and segmentos_correctos)

except AssertionError:
    print(False)
