import polars as pl

# Carica il dataset
df = pl.read_csv("features/orders_all_years_features.csv")

# Colonne dei pattern
pattern_cols = [col for col in df.columns if col.startswith("CDL")]

# Analisi
results = []

for col in pattern_cols:
    subset = df.filter(pl.col(col) > 0)
    if subset.is_empty():
        continue

    total = subset.height
    success = subset.filter(pl.col("Success") == 1).height
    fail = total - success
    success_rate = success / total

    results.append({
        "Pattern": col,
        "Occurrences": total,
        "Successes": success,
        "Fails": fail,
        "Success Rate": round(success_rate, 4)
    })

# Output ordinato per Success Rate
result_df = pl.DataFrame(results).sort("Success Rate", descending=True)
result_df.write_csv("features/pattern_success_rate.csv")
print("📁 Risultato salvato in: features/pattern_success_rate.csv")
print(result_df)
