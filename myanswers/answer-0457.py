entrada, salida_esperada = generar_caso_de_uso_resumen_clientes()
def resumen_clientes(df):
    """
    Solución del caso de uso:
    resume las compras por cliente calculando total gastado,
    total de unidades y ticket promedio.
    """

    resumen = (
        df.groupby("cliente", as_index=False)
          .agg(
              total_gastado=("monto", "sum"),
              total_unidades=("cantidad", "sum"),
              ticket_promedio=("monto", "mean")
          )
          .sort_values("total_gastado", ascending=False)
          .reset_index(drop=True)
    )

    return resumen


resultado = resumen_clientes(**entrada)

print("=== RESULTADO DE LA FUNCIÓN SOLUCIÓN ===")
print(resultado)

print("\n=== OUTPUT ESPERADO DEL GENERADOR ===")
print(salida_esperada)

print("\n=== COMPROBACIÓN ===")

try:
    pd.testing.assert_frame_equal(
        resultado,
        salida_esperada,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10
    )
    print(True)

except AssertionError as error:
    print(False)
    print(error)
