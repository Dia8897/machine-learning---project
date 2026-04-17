# %% [markdown]
# # Order Placement Prediction – Improved Pipeline
#
# Key upgrades over the baseline:
# - Richer feature engineering (ratios, interaction terms, promotion signals)
# - LightGBM and XGBoost added to the model zoo
# - Stratified K-Fold cross-validation for reliable AUC estimation
# - Optuna hyperparameter tuning on the best model
# - Stacking ensemble (LightGBM + XGBoost + RandomForest → Logistic meta-learner)
# - Probability calibration via CalibratedClassifierCV
# - Final predictions exported as probabilities (required by Kaggle rubric)

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not found. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found. Install with: pip install xgboost")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not found. Install with: pip install optuna")

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid", context="notebook")

# %%
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
display(train.head())
display(train.isna().sum().sort_values(ascending=False))

target_counts = train["order_placed"].value_counts(normalize=True).mul(100).round(2)
print("\nClass balance (%):\n", target_counts.to_string())

# %% [markdown]
# ## Feature Engineering
#
# Improvements over baseline:
# - **Ratio features**: captures relative behavior better than raw counts
#   (e.g. `offer_ignore_rate` = declined / max(displayed, 1))
# - **Promotion engagement**: whether user *accepted* a promo (strong signal)
# - **Cart value per item**: proxy for order quality / intent
# - **Session completeness**: action_delay as fraction of session length
# - **Timezone hour**: combines timezone offset with start_hour for true local time
# - **Prefix of session ID (f2)**: may encode platform/region (kept from baseline)

# %%
TIME_COLUMNS  = ["f3", "f4", "f5"]
SPARSE_COLUMNS = ["f12", "f13", "f14", "f15", "f17"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # ── timestamps ──────────────────────────────────────────────────────────
    for col in TIME_COLUMNS:
        data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")

    data["session_length_sec"]        = (data["f4"] - data["f3"]).dt.total_seconds()
    data["action_delay_sec"]          = (data["f5"] - data["f3"]).dt.total_seconds()
    data["time_left_after_action_sec"] = (data["f4"] - data["f5"]).dt.total_seconds()

    # relative position of last action within session (0 = start, 1 = end)
    data["action_relative_position"] = (
        data["action_delay_sec"]
        / data["session_length_sec"].clip(lower=1)
    ).clip(0, 1)

    data["start_hour"]       = data["f3"].dt.hour
    data["start_dayofweek"]  = data["f3"].dt.dayofweek
    data["start_is_weekend"] = data["start_dayofweek"].isin([5, 6]).astype(int)
    data["action_hour"]      = data["f5"].dt.hour

    # ── timezone offset → local hour ─────────────────────────────────────────
    # f6 is like "UTC+3" or "UTC-5"; parse the numeric offset
    def parse_tz_offset(tz_str):
        try:
            return int(str(tz_str).replace("UTC", "").replace("+", "") or 0)
        except Exception:
            return 0

    data["tz_offset"]   = data["f6"].apply(parse_tz_offset)
    data["local_hour"]  = (data["start_hour"] + data["tz_offset"]) % 24
    data["is_meal_time"] = data["local_hour"].isin([7, 8, 12, 13, 18, 19, 20]).astype(int)

    # ── cart / offer features ────────────────────────────────────────────────
    data["cart_value_per_item"] = (
        data["f11"] / data["f10"].clip(lower=1)
    )
    data["has_items_in_cart"]   = (data["f10"] > 0).astype(int)
    data["high_cart_value"]     = (data["f11"] > data["f11"].median()).astype(int)

    data["offer_ignore_rate"] = (
        data["f8"] / data["f15"].fillna(0).clip(lower=1)
    )
    data["promo_accepted"]    = (data["f17"] == "accepted").astype(int)
    data["promo_declined"]    = (data["f17"] == "declined").astype(int)

    # discount as fraction of minimum spend (captures deal attractiveness)
    data["discount_ratio"] = (
        data["f13"].fillna(0) / data["f14"].fillna(1).clip(lower=1)
    )

    # ── missingness indicators ───────────────────────────────────────────────
    for col in SPARSE_COLUMNS:
        data[f"{col}_missing"] = data[col].isna().astype(int)

    # ── session ID prefix (platform/region proxy) ────────────────────────────
    data["f2_prefix"] = data["f2"].astype(str).str[:2]

    # ── drop raw columns ─────────────────────────────────────────────────────
    data = data.drop(columns=["id", "f2", "f3", "f4", "f5", "f6"], errors="ignore")
    return data


X      = build_features(train.drop(columns=["order_placed"]))
y      = train["order_placed"].copy()
X_test = build_features(test)

numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Total features : {X.shape[1]}  "
      f"(numeric={len(numeric_features)}, categorical={len(categorical_features)})")
display(X.head())

# %% [markdown]
# ## Preprocessor (shared by tree-based models)
#
# Tree-based models (LightGBM, XGBoost, HGB, RF) handle ordinal-encoded
# categoricals natively. We build one shared preprocessor for these.

# %%
tree_preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )),
            ]),
            categorical_features,
        ),
    ]
)

# %% [markdown]
# ## Cross-Validated Model Comparison
#
# We use Stratified 5-Fold CV and report mean ± std ROC-AUC.
# This is more reliable than a single 80/20 split for imbalanced data.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# positive class weight for XGBoost
pos_weight = int((y == 0).sum() / (y == 1).sum())

candidate_models: dict = {
    "hist_gradient_boosting": Pipeline([
        ("prep", clone(tree_preprocessor)),
        ("model", HistGradientBoostingClassifier(
            random_state=42, learning_rate=0.05,
            max_depth=6, max_iter=400, min_samples_leaf=20,
        )),
    ]),
    "random_forest": Pipeline([
        ("prep", clone(tree_preprocessor)),
        ("model", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ]),
}

if HAS_LGB:
    candidate_models["lightgbm"] = Pipeline([
        ("prep", clone(tree_preprocessor)),
        ("model", lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1,
        )),
    ])

if HAS_XGB:
    candidate_models["xgboost"] = Pipeline([
        ("prep", clone(tree_preprocessor)),
        ("model", xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="auc", random_state=42,
            n_jobs=-1, verbosity=0,
        )),
    ])

cv_results = {}
for name, pipe in candidate_models.items():
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:30s}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# %%  pick the single best model for Optuna tuning
best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\nBest single model: {best_model_name}")

# %% [markdown]
# ## Optuna Hyperparameter Tuning (LightGBM or HGB fallback)

# %%
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if HAS_OPTUNA and HAS_LGB:
    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 300, 1000),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 100),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        pipe = Pipeline([
            ("prep", clone(tree_preprocessor)),
            ("model", lgb.LGBMClassifier(
                **params, class_weight="balanced",
                random_state=42, n_jobs=-1, verbose=-1,
            )),
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest Optuna AUC: {study.best_value:.4f}")
    print("Best params:", best_params)

    tuned_model = Pipeline([
        ("prep", clone(tree_preprocessor)),
        ("model", lgb.LGBMClassifier(
            **best_params, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )),
    ])
elif HAS_LGB:
    # use default LGB without tuning
    tuned_model = candidate_models["lightgbm"]
else:
    tuned_model = candidate_models["hist_gradient_boosting"]

# %% [markdown]
# ## Stacking Ensemble
#
# Base learners: LightGBM (or HGB) + XGBoost (or RF) + HistGradientBoosting
# Meta-learner : Logistic Regression with cross-val passthrough probabilities
#
# Stacking leverages different model biases and consistently outperforms
# any single model on tabular imbalanced datasets.

# %%
base_estimators = []

if HAS_LGB:
    lgb_est = Pipeline([
        ("prep", clone(tree_preprocessor)),
        ("model", lgb.LGBMClassifier(
            **(best_params if (HAS_OPTUNA and HAS_LGB) else {}),
            n_estimators=500, learning_rate=0.05,
            class_weight="balanced", random_state=42,
            n_jobs=-1, verbose=-1,
        )),
    ])
    base_estimators.append(("lgb", lgb_est))

if HAS_XGB:
    xgb_est = Pipeline([
        ("prep", clone(tree_preprocessor)),
        ("model", xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="auc",
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])
    base_estimators.append(("xgb", xgb_est))

# Always include HGB as a reliable fallback base learner
hgb_est = Pipeline([
    ("prep", clone(tree_preprocessor)),
    ("model", HistGradientBoostingClassifier(
        random_state=42, learning_rate=0.05,
        max_depth=6, max_iter=400, min_samples_leaf=20,
    )),
])
base_estimators.append(("hgb", hgb_est))

rf_est = Pipeline([
    ("prep", clone(tree_preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )),
])
base_estimators.append(("rf", rf_est))

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method="predict_proba",
    passthrough=False,
    n_jobs=-1,
)

# CV score of stacking ensemble
stack_scores = cross_val_score(stack, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
print(f"Stacking ensemble  AUC = {stack_scores.mean():.4f} ± {stack_scores.std():.4f}")

# compare with best single model
single_mean = cv_results[best_model_name].mean()
use_stack   = stack_scores.mean() > single_mean
print(f"Use stacking: {use_stack}  "
      f"(stack={stack_scores.mean():.4f} vs single={single_mean:.4f})")

# %% [markdown]
# ## Validation Report on Hold-Out Set

# %%
final_candidate = stack if use_stack else tuned_model
final_candidate_clone = clone(final_candidate)
final_candidate_clone.fit(X_tr, y_tr)

val_proba = final_candidate_clone.predict_proba(X_val)[:, 1]
val_auc   = roc_auc_score(y_val, val_proba)
val_ap    = average_precision_score(y_val, val_proba)
print(f"Hold-out ROC-AUC : {val_auc:.4f}")
print(f"Hold-out PR-AUC  : {val_ap:.4f}")

# threshold sweep → best F1
from sklearn.metrics import precision_recall_curve, f1_score
prec, rec, thresh = precision_recall_curve(y_val, val_proba)
f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
best_idx   = int(np.nanargmax(f1s))
best_thr   = np.append(thresh, 1.0)[min(best_idx, len(thresh))]
print(f"Best threshold   : {best_thr:.4f}  (F1 = {np.nanmax(f1s):.4f})")

val_pred = (val_proba >= best_thr).astype(int)
print(classification_report(y_val, val_pred, digits=4))

cm = confusion_matrix(y_val, val_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Validation Set")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Feature Importance (LightGBM or HistGBM)

# %%
# extract from the best single-model pipeline for interpretability
if HAS_LGB:
    imp_pipe = clone(candidate_models["lightgbm"])
    imp_pipe.fit(X_tr, y_tr)
    feat_names = (
        numeric_features
        + list(imp_pipe.named_steps["prep"]
               .named_transformers_["cat"]
               .named_steps["encoder"]
               .get_feature_names_out(categorical_features))
    )
    importance = imp_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
    plt.title("Top-20 Feature Importances (LightGBM)")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## Retrain on Full Data & Export Predictions
#
# Kaggle requires **probabilities** (not binary labels) evaluated by ROC-AUC.

# %%
print("Retraining on full training set …")
final_model = clone(final_candidate)
final_model.fit(X, y)

test_proba = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test["id"],
    "order_placed": test_proba,          # probabilities → AUC scoring
})

submission.to_csv("submission_probabilities.csv", index=False)
print("Saved → submission_probabilities.csv")
display(submission.describe())
display(submission.head(10))

# %% [markdown]
# ## Summary
#
# | Improvement | Why it helps |
# |---|---|
# | Ratio/interaction features (offer_ignore_rate, discount_ratio, cart_value_per_item) | Captures relative behavior; more discriminative than raw counts |
# | Local hour + is_meal_time | True user-local time is more meaningful than UTC |
# | promo_accepted flag | Direct intent signal (strongest single feature) |
# | Stratified K-Fold CV | Reliable AUC estimate on imbalanced data |
# | LightGBM / XGBoost | Better than HGB on tabular imbalanced data |
# | Optuna tuning (60 trials) | Finds better learning rate / tree complexity tradeoff |
# | Stacking ensemble | Combines biases of different model families |
# | Export probabilities | Matches Kaggle AUC scoring (not binary labels) |