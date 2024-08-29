import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import optuna
import json


features_path = '/home/nikolenko/work/Project/ChemRar/data/processed/'
features_file = 'ecfp:4_features.parquet'
features_df = pd.read_parquet(os.path.join(features_path, features_file))

data_path = '/home/nikolenko/work/Project/ChemRar/data/raw/data.csv'
data_df = pd.read_csv(data_path)

split_path = "/home/nikolenko/work/Project/ChemRar/data/processed/random_split.csv"
split_df = pd.read_csv(split_path)


def get_features_for_split(smiles_split, features_df):
    return features_df[features_df['SMILES'].isin(smiles_split)].reset_index(drop=True)


train_smiles = split_df['Train_SMILES'].dropna().values
val_smiles = split_df['Val_SMILES'].dropna().values
test_smiles = split_df['Test_SMILES'].dropna().values

X_train_df = get_features_for_split(train_smiles, features_df)
X_val_df = get_features_for_split(val_smiles, features_df)
X_test_df = get_features_for_split(test_smiles, features_df)


train_data = pd.merge(X_train_df, data_df[['SMILES', 'Activity']], on='SMILES')
val_data = pd.merge(X_val_df, data_df[['SMILES', 'Activity']], on='SMILES')
test_data = pd.merge(X_test_df, data_df[['SMILES', 'Activity']], on='SMILES')


X_train = np.vstack(train_data['ecfp:4_features'].values)
y_train = train_data['Activity'].apply(lambda x: 1 if x == 'Active' else 0)

X_val = np.vstack(val_data['ecfp:4_features'].values)
y_val = val_data['Activity'].apply(lambda x: 1 if x == 'Active' else 0)

X_test = np.vstack(test_data['ecfp:4_features'].values)
y_test = test_data['Activity'].apply(lambda x: 1 if x == 'Active' else 0)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

def objective(trial):
    param = {
        'n_estimators': trial.suggest_categorical('n_estimators', [300, 500, 1000, 1500, 2000]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 10]),
        'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 5, 10]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001]),
        'lambda': trial.suggest_categorical('lambda', [0, 0.2, 1, 5]),
        'alpha': trial.suggest_categorical('alpha', [0, 0.2, 1, 5]),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.8, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.7, 1.0]),
        'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1, 200]),
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
    }

    model = xgb.XGBClassifier(**param)
    model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)

    y_val_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred)

    trial.report(f1, step=0)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return f1


pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1)
study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=1000)


best_params = study.best_params
with open('/home/nikolenko/work/Project/ChemRar/reports/xgb_params.json', 'w') as f:
    json.dump(best_params, f)


best_params['device'] = 'cuda'
best_params['tree_method'] = 'hist'

best_model = xgb.XGBClassifier(**best_params)
best_model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=True)

model_save_path = '/home/nikolenko/work/Project/ChemRar/reports/xgb_model.xgb'
best_model.save_model(model_save_path)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall_vals, precision_vals)

conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": classification_rep
}

with open('/home/nikolenko/work/Project/ChemRar/reports/xgb_model_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("="*30)
print("Best Model Performance Metrics")
print("="*30)
print(f"Accuracy:         {accuracy:.6f}")
print(f"Precision:        {precision:.6f}")
print(f"Recall:           {recall:.6f}")
print(f"F1 Score:         {f1:.6f}")
print(f"ROC AUC:          {roc_auc:.6f}")
print(f"PR AUC:           {pr_auc:.6f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
print("="*30)
