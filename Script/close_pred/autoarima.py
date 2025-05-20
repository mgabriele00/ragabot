import polars as pl

def explode_polars(
    input_parquet: str,
    output_parquet: str,
    n_bars: int = 10
) -> None:
    """
    Per ogni riga (barra) del dataset di input:
    - Replica le successive n_bars righe come "barre future"
    - Mantieni i livelli TP_theoretical e SL_theoretical della riga base
    - Calcola target:
        +1 se close_{t+offset+1} >= TP_theoretical
        -1 se close_{t+offset+1} <= SL_theoretical
         0 altrimenti
    """
    df = pl.read_parquet(input_parquet)
    required = {"TP_theoretical", "SL_theoretical", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mancano colonne: {missing}")

    df = df.with_row_index("base_idx")

    df_base = df.select(["base_idx", "TP_theoretical", "SL_theoretical"])
    df_future = df.rename({"base_idx": "future_idx", "Close": "future_close"})

    offsets = pl.DataFrame({"offset": list(range(1, n_bars + 1))})

    expl = df_base.join(offsets, how="cross")
    expl = expl.with_columns((pl.col("base_idx") + pl.col("offset")).alias("future_idx"))

    # ✅ shift di +1 sul future_idx per ottenere close_{t+offset+1}
    expl = expl.with_columns((pl.col("future_idx") + 1).alias("target_idx"))

    # join con close della riga successiva
    expl = expl.join(df.select(["base_idx", "Close"]).rename({
        "base_idx": "target_idx",
        "Close": "target_close"
    }), on="target_idx", how="left")

    # ✅ target calcolata su close_{t+offset+1}
    expl = expl.with_columns(
        pl.when(pl.col("target_close") >= pl.col("TP_theoretical")).then(1)
          .when(pl.col("target_close") <= pl.col("SL_theoretical")).then(-1)
          .otherwise(0).alias("target")
    )

    expl = expl.drop_nulls()
    expl.write_parquet(output_parquet)
    print(f"✅ Dataset esploso salvato in: {output_parquet}")

if __name__ == '__main__':
    explode_polars('feature_dataset.parquet','exploded_dataset.parquet',n_bars=10)
