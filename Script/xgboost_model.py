import os
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

# === Caricamento dati ===
print("📂 Caricamento features...")
df = pl.read_csv("features/orders_all_years_features.csv")

df_train = df.filter(pl.col("Year") < 2024)
df_test = df.filter(pl.col("Year") == 2024)

X_train = df_train.drop(["PnL", "Success", "Datetime", "Entry Time", "Year"]).to_pandas()
y_train = df_train["Success"].to_numpy()
X_test = df_test.drop(["PnL", "Success", "Datetime", "Entry Time", "Year"]).to_pandas()
y_test = df_test["Success"].to_numpy()

# === Selezione feature importanti con modello semplice ===
print("🚀 Modello iniziale per selezione feature...")
base_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
    random_state=42
)
base_model.fit(X_train, y_train)

importances = base_model.feature_importances_
features = X_train.columns
threshold = 0.01
important_features = features[importances > threshold]

print(f"📌 Features selezionate: {list(important_features)}")

X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

# === Training o caricamento modello migliore ===
model_path = "models/single_combination.model"
os.makedirs("models", exist_ok=True)

if os.path.exists(model_path):
    print("✅ Caricamento modello esistente...")
    best_model = xgb.XGBClassifier()
    best_model.load_model(model_path)
else:
    print("🧠 Training e Grid Search...")
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0]
    }

    grid = GridSearchCV(
        estimator=xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
            random_state=42
        ),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        verbose=1
    )

    grid.fit(X_train_selected, y_train)
    best_model = grid.best_estimator_
    best_model.save_model(model_path)
    print(f"💾 Modello salvato in: {model_path}")
    print("📊 Best params:", grid.best_params_)

# === Valutazione ===
print("\n📊 Report classificazione:")
y_pred = best_model.predict(X_test_selected)
y_prob = best_model.predict_proba(X_test_selected)[:, 1]

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# === Precision / Recall curve ===
prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, prec[:-1], label="Precision")
plt.plot(thresholds, rec[:-1], label="Recall")
plt.title("Precision / Recall Curve")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# === Statistiche per soglie di probabilità ===
print("\n📈 Statistiche di classificazione per diversi threshold:")
thresholds_to_check = np.arange(0.5, 0.96, 0.05)
total_positives = sum(y_test)
total_orders = len(y_test)

results = []
for thresh in thresholds_to_check:
    mask = y_prob >= thresh
    selected = mask.sum()
    if selected == 0:
        continue
    true_positives = (y_test[mask] == 1).sum()
    precision = true_positives / selected
    recall = true_positives / total_positives
    coverage = selected / total_orders
    results.append({
        "Threshold": round(thresh, 2),
        "Ordini Fatti": selected,
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "Coverage (%)": round(coverage * 100, 2)
    })

stats_df = pd.DataFrame(results)
print(stats_df)

# === Grafico riassuntivo ===
plt.figure(figsize=(12, 6))
plt.plot(stats_df["Threshold"], stats_df["Precision (%)"], label="Precision (%)", marker='o')
plt.plot(stats_df["Threshold"], stats_df["Recall (%)"], label="Recall (%)", marker='o')
plt.plot(stats_df["Threshold"], stats_df["Coverage (%)"], label="Coverage (%)", marker='o', linestyle='--')
plt.title("📊 Precision / Recall / Coverage vs Threshold")
plt.xlabel("Threshold di probabilità")
plt.ylabel("Percentuale (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === (Opzionale) Salvataggio delle metriche ===
stats_df.to_csv("models/threshold_metrics.csv", index=False)
print("📄 Statistiche salvate in models/threshold_metrics.csv")
