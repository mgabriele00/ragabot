import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, make_scorer, f1_score
)
from sklearn.utils import resample
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import optuna

# 1. Carica e pulisci dataset
df = pd.read_parquet('../close_pred/exploded_dataset.parquet')
df = df.dropna()
print(f"Righe totali: {len(df)}")

# 2. Encoding target
label_map = {-1: 0, 0: 1, 1: 2}
df['y'] = df['target'].map(label_map)

# 3. Bilanciamento con undersampling
df_0 = df[df['y'] == 0]
df_1 = df[df['y'] == 1]
df_2 = df[df['y'] == 2]
n_min = min(len(df_0), len(df_2))
df_1_under = resample(df_1, replace=False, n_samples=n_min, random_state=42)
df_balanced = pd.concat([df_0, df_1_under, df_2], ignore_index=True)
print(f"Bilanciato: {df_balanced['y'].value_counts().to_dict()}")

# 4. Feature selection iniziale
exclude_cols = [col for col in ['datetime', 'row_nr', 'offset', 'target', 'y'] if col in df_balanced.columns]
X_full = df_balanced.drop(columns=exclude_cols)
y = df_balanced['y']
X_full = X_full.astype(np.float32)

# 5. Primo training per importance
X_sample, _, y_sample, _ = train_test_split(X_full, y, train_size=300_000, stratify=y, random_state=42)
model_temp = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1
)
model_temp.fit(X_sample, y_sample)

importances = model_temp.feature_importances_
important_features = X_full.columns[importances > 0.0025]
print(f"Feature selezionate: {len(important_features)} su {X_full.shape[1]}")
pd.Series(important_features).to_csv("selected_features.csv", index=False)

X = X_full[important_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 6. Optuna tuning
def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'random_state': 42,
        'n_jobs': -1
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='macro')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)
print("Best trial:", study.best_trial.params)

# 7. Train finale con best params
final_model = XGBClassifier(
    **study.best_trial.params,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train, y_train)

# 8. Probabilità e threshold
probs = final_model.predict_proba(X_test)

def get_best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1[:-1])
    return thresholds[best_idx], precision[best_idx], recall[best_idx], f1[best_idx]

best_thresh_0, *_ = get_best_f1_threshold(y_test == 0, probs[:, 0])
best_thresh_2, *_ = get_best_f1_threshold(y_test == 2, probs[:, 2])

y_pred_opt = np.where(probs[:, 0] >= best_thresh_0, 0,
                      np.where(probs[:, 2] >= best_thresh_2, 2, 1))
inv_map = {0: -1, 1: 0, 2: 1}
y_test_labels = y_test.map(inv_map)
y_pred_labels = pd.Series(y_pred_opt).map(inv_map)

print("\n=== REPORT THRESHOLD OTTIMALI ===")
print(confusion_matrix(y_test_labels, y_pred_labels, labels=[-1, 0, 1]))
print(classification_report(y_test_labels, y_pred_labels, target_names=["SL_hit", "No_hit", "TP_hit"]))

# 9. Salva modello finale
joblib.dump({
    'model': final_model,
    'thresh_sl_opt': best_thresh_0,
    'thresh_tp_opt': best_thresh_2,
    'selected_features': list(important_features),
    'best_params': study.best_trial.params
}, 'xgb_tp_sl_model_thresh.joblib')

print("Modello, threshold e feature salvati in xgb_tp_sl_model_thresh.joblib e selected_features.csv")
