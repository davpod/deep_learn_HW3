import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ======================
# CONFIG
# ======================
TRAIN_CSV = "train.csv"
TEST_LABELS_CSV = "test_labels.csv"  # merged test + relevance

# ======================
# LOAD DATA
# ======================
train_df = pd.read_csv(TRAIN_CSV, encoding="latin1")
test_df = pd.read_csv(TEST_LABELS_CSV, encoding="latin1")  # contains relevance

# ======================
# TRAIN / VALID SPLIT
# ======================
train_text = train_df["search_term"].astype(str) + " || " + train_df["product_title"].astype(str)
X_tr_text, X_val_text, y_tr, y_val_split = train_test_split(
    train_text, train_df["relevance"], test_size=0.2, random_state=42
)

# ======================
# TRAIN BASELINE MODEL
# ======================
print("Training baseline model...")
start_time_base = time.time()

# CountVectorizer on characters
vectorizer = CountVectorizer(analyzer="char", ngram_range=(3,5), min_df=5)
X_tr = vectorizer.fit_transform(X_tr_text)
X_val = vectorizer.transform(X_val_text)

# Ridge regression
baseline = Ridge(alpha=1.0)
baseline.fit(X_tr, y_tr)

# --- Training metrics ---
train_preds_base = baseline.predict(X_tr)
train_rmse_base = np.sqrt(mean_squared_error(y_tr, train_preds_base))
train_mae_base = mean_absolute_error(y_tr, train_preds_base)
print(f"Baseline Train RMSE: {train_rmse_base:.4f} | MAE: {train_mae_base:.4f}")

# ======================
# VALIDATION METRICS
# ======================
val_preds_base = baseline.predict(X_val)
val_rmse_base = np.sqrt(mean_squared_error(y_val_split, val_preds_base))
val_mae_base = mean_absolute_error(y_val_split, val_preds_base)
print(f"Baseline Validation RMSE: {val_rmse_base:.4f} | MAE: {val_mae_base:.4f}")

# ======================
# TEST METRICS
# ======================
test_text = test_df["search_term"].astype(str) + " || " + test_df["product_title"].astype(str)
test_preds_base = baseline.predict(vectorizer.transform(test_text))
test_rmse_base = np.sqrt(mean_squared_error(test_df["relevance"], test_preds_base))
test_mae_base = mean_absolute_error(test_df["relevance"], test_preds_base)

baseline_runtime = time.time() - start_time_base
print(f"Baseline Test RMSE: {test_rmse_base:.4f} | MAE: {test_mae_base:.4f}")
print(f"Total Baseline Runtime (Train + Inference): {baseline_runtime:.2f} sec")
