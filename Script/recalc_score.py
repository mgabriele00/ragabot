import polars as pl

# Percorso al tuo CSV salvato
csv_file = "../results/strategies_by_params_all_years.csv"

# 1) Carica il CSV
df = pl.read_csv(csv_file)

# 2) Trova le colonne equity_YYYY e drawdown_YYYY
equity_cols = [col for col in df.columns if col.startswith("equity_")]
years = [int(col.split("_")[1]) for col in equity_cols]
drawdown_cols = [f"drawdown_{y}" for y in years]

# 3) Definisci i pesi
alpha, beta, gamma = 0.9, 0.05, 0.05

# 4) Calcola mean_equity, mean_drawdown e mean_squared_equity
df = df.with_columns([
    pl.mean_horizontal(pl.col(equity_cols)).alias("mean_equity"),
    pl.mean_horizontal(pl.col(drawdown_cols)).alias("mean_drawdown"),
    pl.mean_horizontal([pl.col(c)**2 for c in equity_cols]).alias("mean_squared_equity"),
])

# 5) Calcola varianza, deviazione standard e volatilità relativa
df = df.with_columns([
    (pl.col("mean_squared_equity") - pl.col("mean_equity")**2).alias("var_equity"),
    pl.sqrt(pl.col("mean_squared_equity") - pl.col("mean_equity")**2).alias("std_equity"),
    (pl.col("std_equity") / pl.col("mean_equity")).alias("rel_volatility"),
])

# 6) Calcola lo score
df = df.with_columns([
    (alpha * pl.col("mean_equity")
     - beta  * pl.col("mean_drawdown")
     - gamma * pl.col("rel_volatility")).alias("score")
])

# 7) Estrai la riga con lo score migliore
best = df.sort("score", descending=True).head(1)

print("Strategia migliore:")
print(best)
