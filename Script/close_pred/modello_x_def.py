import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# === 1. Carica dati e feature selezionate
df = pd.read_parquet('../close_pred/exploded_dataset.parquet')
df = df.dropna()
features = pd.read_csv("selected_features.csv", header=None).squeeze().tolist()

# Escludi colonne tecniche o target
exclude = ['0', 'base_idx', 'base_idx_right', 'future_idx', 'target_idx', 'target_close']
features = [f for f in features if f not in exclude]

# Encoding target
label_map = {-1: 0, 0: 1, 1: 2}
df['y'] = df['target'].map(label_map)

# Prepara X e y
X = df[features].astype(np.float32)
y = df['y']

# Split stratificato
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# === 2. Parametri migliori da Optuna
best_params = {
    'learning_rate': 0.19866752132229515,
    'max_depth': 7,
    'n_estimators': 400,
    'subsample': 0.8008329728063355,
    'colsample_bytree': 0.6475306633196563
}

# === 3. Allena modello finale
model = XGBClassifier(
    **best_params,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# === 4. ROC per ogni classe
y_score = model.predict_proba(X_test)
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# === 5. Plot ROC
plt.figure(figsize=(10, 6))
colors = ['darkred', 'gray', 'green']
labels = {-1: "SL_hit", 0: "No_hit", 1: "TP_hit"}
inv_map = {0: -1, 1: 0, 2: 1}

for i in range(n_classes):
    label = labels[inv_map[i]]
    plt.plot(fpr[i], tpr[i], label=f"Class {label} (AUC = {roc_auc[i]:.3f})", linewidth=2, color=colors[i])

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve per XGBoost - Classi TP / SL / No Hit")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
