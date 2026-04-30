import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, ConfusionMatrixDisplay)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ── LOAD ──────────────────────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv")
y = train["order_placed"].astype(int)

# ── FEATURE ENGINEERING (identical to triallastone) ───────────────────────────
def make_features(df):
    df = df.copy()
    for c in ["f3", "f4", "f5"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["session_duration"] = (df["f4"] - df["f3"]).dt.total_seconds()
    df["active_time"]      = (df["f5"] - df["f3"]).dt.total_seconds()
    df["idle_time"]        = (df["f4"] - df["f5"]).dt.total_seconds()
    df["date"]  = df["f3"].dt.strftime("%Y-%m-%d")
    df["hour"]  = df["f3"].dt.hour
    df["dow"]   = df["f3"].dt.dayofweek
    df["day"]   = df["f3"].dt.day
    df["hour_sin"] = np.sin(2*np.pi*df["hour"].fillna(0)/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"].fillna(0)/24)
    df["id_mod_2"]   = df["id"] % 2
    df["id_mod_5"]   = df["id"] % 5
    df["id_mod_10"]  = df["id"] % 10
    df["id_mod_100"] = df["id"] % 100
    df["has_cart"]  = (df["f10"] > 0).astype(int)
    df["has_value"] = (df["f11"] > 0).astype(int)
    df["accepted"]  = (df["f17"].astype(str) == "ACCEPTED").astype(int)
    df["ignored"]   = (df["f17"].astype(str) == "IGNORED").astype(int)
    df["declined"]  = (df["f17"].astype(str) == "DECLINED").astype(int)
    df["conversion_signal"] = df["has_cart"] + df["has_value"] + df["accepted"]
    df["meets_min"]   = (df["f11"] >= df["f14"]).astype(int)
    df["final_intent"] = (df["meets_min"] + df["accepted"]
                          + (df["f10"] > 2).astype(int)
                          + (df["f13"] > 0).astype(int))
    df["cart_quality"]     = df["f11"] / (df["f10"] + 1)
    df["cart_to_min"]      = df["f11"] / (df["f14"] + 1)
    df["discount_to_cart"] = df["f13"] / (df["f11"] + 1)
    df["discount_to_min"]  = df["f13"] / (df["f14"] + 1)
    df["offers_per_decline"] = df["f15"] / (df["f8"] + 1)
    df["items_per_offer"]    = df["f10"] / (df["f15"] + 1)
    df["value_per_offer"]    = df["f11"] / (df["f15"] + 1)
    df["urgency"]      = df["f10"] / (df["session_duration"] + 1)
    df["value_speed"]  = df["f11"] / (df["session_duration"] + 1)
    df["promo_resp"]  = df["f12"].astype(str) + "_" + df["f17"].astype(str)
    df["cust_promo"]  = df["f9"].astype(str)  + "_" + df["f12"].astype(str)
    df["cust_resp"]   = df["f9"].astype(str)  + "_" + df["f17"].astype(str)
    df["action_resp"] = df["f7"].astype(str)  + "_" + df["f17"].astype(str)
    df["date_promo"]  = df["date"].astype(str) + "_" + df["f12"].astype(str)
    df["date_resp"]   = df["date"].astype(str) + "_" + df["f17"].astype(str)
    cat_cols = ["f6","f7","f9","f12","f17","date",
                "promo_resp","cust_promo","cust_resp",
                "action_resp","date_promo","date_resp"]
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("missing")
    df = df.drop(columns=["f2","f3","f4","f5"], errors="ignore")
    return df

X = make_features(train.drop(columns=["order_placed"]))

cat_cols = ["f6","f7","f9","f12","f17","date",
            "promo_resp","cust_promo","cust_resp",
            "action_resp","date_promo","date_resp"]
cat_cols = [c for c in cat_cols if c in X.columns]

# ── TARGET ENCODING ───────────────────────────────────────────────────────────
def add_cv_target_encoding(X, y, cols, n_splits=5):
    X = X.copy()
    global_mean = y.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for col in cols:
        te_name = col + "_te"
        X[te_name] = global_mean
        for tr_idx, val_idx in skf.split(X, y):
            tmp = pd.DataFrame({"col": X.iloc[tr_idx][col], "target": y.iloc[tr_idx]})
            means = tmp.groupby("col")["target"].mean()
            X.loc[X.index[val_idx], te_name] = X.iloc[val_idx][col].map(means).fillna(global_mean)
    return X

te_cols = ["date","f7","f9","f12","f17",
           "promo_resp","cust_promo","cust_resp","date_promo","date_resp"]
te_cols = [c for c in te_cols if c in X.columns]
X = add_cv_target_encoding(X, y, te_cols)
print("Features ready:", X.shape)

# ── TRAIN / VAL SPLIT ─────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {X_tr.shape}, Val: {X_val.shape}")
print(f"Positive rate — train: {y_tr.mean():.4f}, val: {y_val.mean():.4f}")

# ── LIGHTGBM ──────────────────────────────────────────────────────────────────
print("\nTraining LightGBM...")
X_tr_lgb  = X_tr.copy()
X_val_lgb = X_val.copy()
for c in cat_cols:
    X_tr_lgb[c]  = X_tr_lgb[c].astype("category")
    X_val_lgb[c] = X_val_lgb[c].astype("category")

lgb = LGBMClassifier(
    objective="binary", metric="auc",
    n_estimators=2500, learning_rate=0.018,
    num_leaves=96, min_child_samples=45,
    subsample=0.88, colsample_bytree=0.88,
    reg_alpha=0.3, reg_lambda=1.5,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb.fit(X_tr_lgb, y_tr)
prob_lgb = lgb.predict_proba(X_val_lgb)[:, 1]
print(f"LGB Val AUC: {roc_auc_score(y_val, prob_lgb):.4f}")

# ── CATBOOST ──────────────────────────────────────────────────────────────────
print("\nTraining CatBoost...")
cat = CatBoostClassifier(
    iterations=2500, learning_rate=0.018,
    depth=7, l2_leaf_reg=6,
    loss_function="Logloss", eval_metric="AUC",
    random_seed=42, verbose=False
)
cat.fit(X_tr, y_tr, cat_features=cat_cols,
        eval_set=(X_val, y_val), use_best_model=True, verbose=False)
prob_cat = cat.predict_proba(X_val)[:, 1]
print(f"CAT Val AUC: {roc_auc_score(y_val, prob_cat):.4f}")

# ── BLEND 51/49 ───────────────────────────────────────────────────────────────
prob_blend = 0.51 * prob_lgb + 0.49 * prob_cat
print(f"\nBlend Val AUC (51% LGB + 49% CAT): {roc_auc_score(y_val, prob_blend):.4f}")

# ── CONFUSION MATRIX AT THRESHOLD 0.25 ───────────────────────────────────────
threshold = 0.25
y_pred = (prob_blend >= threshold).astype(int)

cm = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*50}")
print(f"CONFUSION MATRIX (threshold = {threshold})")
print(f"{'='*50}")
print(f"                 Predicted NO   Predicted YES")
print(f"Actual NO          {tn:>8}        {fp:>8}")
print(f"Actual YES         {fn:>8}        {tp:>8}")
print(f"\nTrue Negatives  (TN): {tn:,}")
print(f"False Positives (FP): {fp:,}  <- predicted order but didn't place")
print(f"False Negatives (FN): {fn:,}  <- missed actual orders")
print(f"True Positives  (TP): {tp:,}  <- correctly caught orders")
print(f"\nPrecision : {tp/(tp+fp):.4f}")
print(f"Recall    : {tp/(tp+fn):.4f}")
print(f"F1 Score  : {2*tp/(2*tp+fp+fn):.4f}")
print()
print(classification_report(y_val, y_pred, target_names=["No Order","Order Placed"], digits=4))

# ── SAVE PLOT ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Order", "Order Placed"])
disp.plot(ax=ax, colorbar=False, cmap="Greens")
ax.set_title(f"Confusion Matrix — Final Model (LGB 51% + CatBoost 49%)\nThreshold = {threshold}  |  Val AUC = {roc_auc_score(y_val, prob_blend):.4f}", fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix_final.png", dpi=150)
print("Saved: confusion_matrix_final.png")
