import os
import polars as pl
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import numpy as np
import pandas as pd

# === Caricamento dati ===
print("📂 Caricamento features...")
df = pl.read_csv("features/orders_all_years_features.csv")

df_train = df.filter(pl.col("Year") < 2024)
df_test = df.filter(pl.col("Year") == 2024)

X_train = df_train.drop(["PnL", "Success", "Datetime", "Entry Time", "Year"]).to_pandas()
y_train = df_train["Success"].to_numpy()
X_test = df_test.drop(["PnL", "Success", "Datetime", "Entry Time", "Year"]).to_pandas()
y_test = df_test["Success"].to_numpy()

# === XGBoost iniziale per valutare importanza ===
print("🚀 Modello iniziale per valutare importanza feature...")
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=(sum(y_train==0)/sum(y_train==1)),
    random_state=42
)
model.fit(X_train, y_train)

# === Selezione feature più importanti ===
importances = model.feature_importances_
features = X_train.columns
threshold = 0.01  # soglia minima di importanza

important_features = features[importances > threshold]
print(f"📌 Features selezionate: {list(important_features)}")

X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

# === Grid Search Estesa ===
print("🔍 Grid Search estesa con feature selezionate...")
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
        scale_pos_weight=(sum(y_train==0)/sum(y_train==1)),
        random_state=42
    ),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    verbose=1
)

grid.fit(X_train_selected, y_train)

# === Valutazione ===
best_model = grid.best_estimator_
print("✅ Miglior modello:")
print(grid.best_params_)

y_pred = best_model.predict(X_test_selected)
y_prob = best_model.predict_proba(X_test_selected)[:, 1]

print("\n📊 Report classificazione:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# === Precision / Recall ===
prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(10,6))
plt.plot(thresholds, prec[:-1], label="Precision")
plt.plot(thresholds, rec[:-1], label="Recall")
plt.title("Precision / Recall Curve")
plt.grid()
plt.legend()
plt.show()

# === Feature Importances finali ===
final_importances = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(important_features, final_importances)
plt.title("🔍 Feature Importances (finale)")
plt.tight_layout()
plt.show()


# === Salvataggio del modello XGBoost ===
model_path = "models/single_combination.model"
os.makedirs("models", exist_ok=True)
best_model.save_model(model_path)
print(f"💾 Modello salvato in: {model_path}")
