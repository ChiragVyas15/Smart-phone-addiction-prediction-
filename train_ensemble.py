import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def build_features(df):
    out = df.copy()
    
    # Missing count feature
    out['missing_count'] = df.isnull().sum(axis=1)
    
    # Numerical features
    screen = out['daily_screen_time_hours']
    social = out['social_media_hours']
    gaming = out['gaming_hours']
    work = out['work_study_hours']
    sleep = out['sleep_hours']
    opens = out['app_opens_per_day']
    notifs = out['notifications_per_day']
    weekend = out['weekend_screen_time']
    age = out['age']
    
    # Domain Ratios and Engineered Features
    out['total_screen_activity'] = screen + social + gaming
    out['weekend_extra_screen'] = weekend - screen
    out['screen_sleep_ratio'] = screen / (sleep + 0.1)
    out['social_screen_ratio'] = social / (screen + 0.1)
    out['gaming_screen_ratio'] = gaming / (screen + 0.1)
    out['work_screen_ratio'] = work / (screen + 0.1)
    out['notification_app_ratio'] = notifs / (opens + 1.0)
    out['notifications_per_screen_hour'] = notifs / (screen + 0.1)
    out['app_opens_per_screen_hour'] = opens / (screen + 0.1)
    out['screen_minus_sleep'] = screen - sleep
    out['screen_work_total'] = screen + work
    out['free_time_estimate'] = 24.0 - (sleep.fillna(7) + screen.fillna(4) + work.fillna(4))
    out['social_gaming_ratio'] = social / (gaming + 0.1)
    
    out['leisure_screen'] = social.fillna(0) + gaming.fillna(0)
    out['leisure_screen_ratio'] = out['leisure_screen'] / (screen + 0.1)
    out['age_screen_interaction'] = age * screen
    
    return out

def main():
    root = Path('.').resolve()
    train_path = root / 'train.csv'
    test_path = root / 'test.csv'
    
    log(f"Loading data from {train_path.name} and {test_path.name}...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    log(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    y = train['addicted_label'].astype(int)
    test_ids = test['id']
    
    # Prepare features
    log("Building features...")
    df_all = pd.concat([train.drop(columns=['addicted_label']), test], ignore_index=True)
    df_all_feats = build_features(df_all)
    
    # Categoricals
    cat_cols = ['gender', 'stress_level', 'academic_work_impact']
    for col in cat_cols:
        df_all_feats[col] = df_all_feats[col].astype('category')
        
    X = df_all_feats.iloc[:len(train)].drop(columns=['id'])
    X_test = df_all_feats.iloc[len(train):].drop(columns=['id'])
    
    log(f"Prepared {X.shape[1]} features.")
    
    # K-Fold Strategy
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(train))
    oof_cb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))
    
    preds_lgb = np.zeros(len(test))
    preds_cb = np.zeros(len(test))
    preds_xgb = np.zeros(len(test))
    
    # For CatBoost format
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str).fillna('missing')
        X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')
        
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        log(f"\n--- Training Fold {fold+1}/{N_SPLITS} ---")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. LightGBM
        log("Fitting LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.04,
            num_leaves=127,
            max_depth=9,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        val_p = lgb_model.predict_proba(X_va)[:, 1]
        oof_lgb[val_idx] = val_p
        preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / N_SPLITS
        log(f"Fold {fold+1} LGBM AUC: {roc_auc_score(y_va, val_p):.5f}, Accuracy: {accuracy_score(y_va, (val_p >= 0.5).astype(int)):.5f}")
        
        # 2. CatBoost
        log("Fitting CatBoost...")
        X_tr_cb, X_va_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
        cb_model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.05,
            depth=8,
            eval_metric='Logloss',
            random_seed=42 + fold,
            thread_count=-1,
            verbose=0
        )
        cb_model.fit(
            X_tr_cb, y_tr,
            cat_features=cat_cols,
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=50
        )
        val_p = cb_model.predict_proba(X_va_cb)[:, 1]
        oof_cb[val_idx] = val_p
        preds_cb += cb_model.predict_proba(X_test_cb)[:, 1] / N_SPLITS
        log(f"Fold {fold+1} CatBoost AUC: {roc_auc_score(y_va, val_p):.5f}, Accuracy: {accuracy_score(y_va, (val_p >= 0.5).astype(int)):.5f}")
        
        # 3. XGBoost
        log("Fitting XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.04,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            enable_categorical=True,
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        val_p = xgb_model.predict_proba(X_va)[:, 1]
        oof_xgb[val_idx] = val_p
        preds_xgb += xgb_model.predict_proba(X_test)[:, 1] / N_SPLITS
        log(f"Fold {fold+1} XGBoost AUC: {roc_auc_score(y_va, val_p):.5f}, Accuracy: {accuracy_score(y_va, (val_p >= 0.5).astype(int)):.5f}")
        
    log("\n=== Cross-Validation Results ===")
    log(f"LGBM OOF Accuracy: {accuracy_score(y, (oof_lgb >= 0.5).astype(int)):.5f}, ROC-AUC: {roc_auc_score(y, oof_lgb):.5f}")
    log(f"CatBoost OOF Accuracy: {accuracy_score(y, (oof_cb >= 0.5).astype(int)):.5f}, ROC-AUC: {roc_auc_score(y, oof_cb):.5f}")
    log(f"XGBoost OOF Accuracy: {accuracy_score(y, (oof_xgb >= 0.5).astype(int)):.5f}, ROC-AUC: {roc_auc_score(y, oof_xgb):.5f}")
    
    # Blending
    oof_ens = 0.40 * oof_lgb + 0.35 * oof_cb + 0.25 * oof_xgb
    preds_ens = 0.40 * preds_lgb + 0.35 * preds_cb + 0.25 * preds_xgb
    
    # Threshold search
    best_thresh = 0.5
    best_acc = accuracy_score(y, (oof_ens >= 0.5).astype(int))
    for t in np.linspace(0.35, 0.65, 61):
        acc = accuracy_score(y, (oof_ens >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
            
    log(f"\n=== ENSEMBLE RESULTS ===")
    log(f"Ensemble OOF ROC-AUC: {roc_auc_score(y, oof_ens):.5f}")
    log(f"Ensemble OOF LogLoss: {log_loss(y, oof_ens):.5f}")
    log(f"Default Threshold (0.50) Accuracy: {accuracy_score(y, (oof_ens >= 0.5).astype(int)):.5f}")
    log(f"Optimal Threshold ({best_thresh:.4f}) Highest Accuracy: {best_acc:.5f}")
    
    # Save submission files
    sub = pd.DataFrame({'id': test_ids, 'addicted_label': preds_ens})
    sub.to_csv('submission.csv', index=False)
    log(f"Saved submission.csv ({len(sub)} rows)")
    
    sub_binary = pd.DataFrame({'id': test_ids, 'addicted_label': (preds_ens >= best_thresh).astype(int)})
    sub_binary.to_csv('final_submission.csv', index=False)
    log(f"Saved final_submission.csv with binary predictions using optimal threshold {best_thresh:.4f}")
    
    # Save metrics summary
    np.savez(
        'model_metrics.npz',
        y_true=y.values,
        oof_ens=oof_ens,
        oof_lgb=oof_lgb,
        oof_cb=oof_cb,
        oof_xgb=oof_xgb,
        preds_ens=preds_ens,
        best_thresh=best_thresh,
        best_acc=best_acc
    )
    log("Saved model_metrics.npz for evaluation and plotting.")

if __name__ == '__main__':
    main()
