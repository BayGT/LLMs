def resumen_clientes(df):
    """
    Solución del caso de uso:
    resume las ventas por cliente calculando el total gastado,
    el total de unidades compradas y el ticket promedio.
    """

    # 1. Agrupar por cliente
    resumen = (
        df.groupby("cliente", as_index=False)
          .agg(
              total_gastado=("monto", "sum"),
              total_unidades=("cantidad", "sum"),
              ticket_promedio=("monto", "mean")
          )
    )

    # 2. Ordenar de mayor a menor según el total gastado
    resumen = resumen.sort_values("total_gastado", ascending=False)

    # 3. Reiniciar el índice
    resumen = resumen.reset_index(drop=True)

    # 4. Devolver el DataFrame final
    return resumen


# Comprobación de la función solución usando el caso de uso generado

resultado = resumen_clientes(**entrada)

print("=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print(resultado)

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")
print(salida_esperada)

print("\n=== COMPROBACIÓN ===")

try:
    pd.testing.assert_frame_equal(resultado, salida_esperada)
    print(True)
except AssertionError:
    print(False)
