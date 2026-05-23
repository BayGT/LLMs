import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def evaluar_lift_por_deciles(df, score_col, target_col, n_bins=5):
    data = df.copy()

    tasa_global = data[target_col].mean()

    segmentos = pd.qcut(
        data[score_col],
        q=n_bins,
        duplicates="drop"
    )

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

    tabla_lift["lift"] = tabla_lift["tasa_positivos"] / tasa_global

    tabla_lift = (
        tabla_lift
        .sort_values("score_max", ascending=False)
        .reset_index(drop=True)
    )

    tabla_lift["segmento"] = np.arange(1, len(tabla_lift) + 1)

    tabla_lift = tabla_lift[
        ["segmento", "score_min", "score_max", "n", "tasa_positivos", "lift"]
    ].copy()

    auc = float(roc_auc_score(data[target_col], data[score_col]))

    return {
        "tabla_lift": tabla_lift,
        "auc": auc,
        "n_segmentos": int(len(tabla_lift))
    }
