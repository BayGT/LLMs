import pandas as pd

def resumen_clientes(df):
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
