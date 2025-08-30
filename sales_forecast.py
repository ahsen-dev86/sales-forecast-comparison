# sales_forecasting.py
import os
import sys
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# optional imports (gracefully handle if not installed)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
plt.style.use('default')

# ---------------- CONFIG ----------------
DATA_DIR = 'data'  # folder containing train.csv, features.csv, stores.csv
SALES_CSV = os.path.join(DATA_DIR, 'train.csv')
FEATURES_CSV = os.path.join(DATA_DIR, 'features.csv')
STORES_CSV = os.path.join(DATA_DIR, 'stores.csv')

# ---------------- LOAD ----------------
for fp in [SALES_CSV, FEATURES_CSV, STORES_CSV]:
    if not os.path.exists(fp):
        print(f'ERROR: required file not found: {fp}')
        print('Put train.csv, features.csv and stores.csv into the data/ folder and re-run.')
        sys.exit(1)

sales = pd.read_csv(SALES_CSV, parse_dates=['Date'])
feat = pd.read_csv(FEATURES_CSV, parse_dates=['Date'])
stores = pd.read_csv(STORES_CSV)

# Normalize obvious column name casing differences (if any)
if 'weekly_sales' in sales.columns and 'Weekly_Sales' not in sales.columns:
    sales.rename(columns={'weekly_sales': 'Weekly_Sales'}, inplace=True)

# ---------------- MERGE ----------------
df = sales.merge(feat, on=['Store', 'Date'], how='left').merge(stores, on='Store', how='left')
df.sort_values(['Store', 'Dept', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------------- FIX DUPLICATE COLUMNS & MISSING EXTERNALS ----------------
# Handle duplicate IsHoliday columns created by merge
if 'IsHoliday_x' in df.columns and 'IsHoliday_y' in df.columns:
    df['IsHoliday'] = df[['IsHoliday_x', 'IsHoliday_y']].max(axis=1)
    df.drop(columns=['IsHoliday_x', 'IsHoliday_y'], inplace=True)
elif 'IsHoliday' not in df.columns:
    # fallback if the column doesn't exist
    df['IsHoliday'] = 0

# Convert possible boolean/str to int (0/1)
try:
    df['IsHoliday'] = df['IsHoliday'].astype(int)
except Exception:
    # try converting common truthy strings
    df['IsHoliday'] = df['IsHoliday'].map(lambda v: 1 if str(v).lower() in ['true','1','yes'] else 0)

# Ensure external numeric features exist; if missing, create and fill with median
external_numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
for col in external_numeric_cols:
    if col not in df.columns:
        df[col] = np.nan
    # fill missing external numeric values with median (per column)
    if df[col].isnull().any():
        med = df[col].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        df[col].fillna(med, inplace=True)

# ---------------- TIME FEATURES ----------------
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
# .isocalendar().week returns a Series with dtype 'UInt32' in pandas >=1.1; cast to int
df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
df['dayofweek'] = df['Date'].dt.dayofweek
df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
df['quarter'] = df['Date'].dt.quarter

# ---------------- LAGS & ROLLING FEATURES ----------------
lags = [1, 2, 3, 4, 12, 52]
for lag in lags:
    df[f'lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)

rolling_windows = [3, 7, 13]
for w in rolling_windows:
    df[f'rolling_mean_{w}'] = (
        df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=w).mean().reset_index(0, drop=True)
    )
    df[f'rolling_std_{w}'] = (
        df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(1).rolling(window=w).std().reset_index(0, drop=True)
    )

# ---------------- DROP NA (from lag/rolling creation) ----------------
# Keep a copy of how big the df was
initial_rows = len(df)
df.dropna(inplace=True)
dropped = initial_rows - len(df)
print(f'Dropped {dropped} rows due to NA after creating lag/rolling features.')

# ---------------- PREP FEATURES LIST (only include columns present) ----------------
base_time_feats = ['year', 'month', 'weekofyear', 'dayofweek', 'is_month_start', 'is_month_end', 'quarter', 'IsHoliday']
external_feats_present = [c for c in external_numeric_cols if c in df.columns]
lag_feats = [f'lag_{l}' for l in lags if f'lag_{l}' in df.columns]
rolling_feats = [f'rolling_mean_{w}' for w in rolling_windows if f'rolling_mean_{w}' in df.columns] + \
                [f'rolling_std_{w}' for w in rolling_windows if f'rolling_std_{w}' in df.columns]

features = base_time_feats + external_feats_present + lag_feats + rolling_feats

# Final sanity check: ensure all features exist
features = [f for f in features if f in df.columns]
print(f'Using {len(features)} features. Example features: {features[:10]}')

# ---------------- TRAIN/TEST SPLIT (time-aware) ----------------
cutoff_date = df['Date'].quantile(0.8)
train = df[df['Date'] <= cutoff_date].copy()
test = df[df['Date'] > cutoff_date].copy()

X_train, y_train = train[features], train['Weekly_Sales']
X_test, y_test = test[features], test['Weekly_Sales']

print(f'Train rows: {len(X_train)}, Test rows: {len(X_test)}')

# ---------------- MODELS ----------------
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0),
}
if XGB_AVAILABLE:
    models['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0)
else:
    print('xgboost not installed or failed import — skipping XGBoost model.')
if LGB_AVAILABLE:
    models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=0)
else:
    print('lightgbm not installed or failed import — skipping LightGBM model.')

results = {}
for name, model in models.items():
    try:
        print(f'\nTraining {name} ...')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))  # compatible with all sklearn versions
        results[name] = {'rmse': rmse, 'preds': preds, 'model': model}
        print(f'{name} RMSE: {rmse:.4f}')
    except Exception as e:
        print(f'Error training {name}: {e}')

if not results:
    print('No models were trained successfully — exiting.')
    sys.exit(1)

# ---------------- BEST MODEL ----------------
best_name = min(results.keys(), key=lambda k: results[k]['rmse'])
best_rmse = results[best_name]['rmse']
print(f'\nBest model: {best_name} with RMSE = {best_rmse:.4f}')

# ---------------- PLOT: aggregated Actual vs Predicted (by Date) ----------------
# Build dataframe with predictions of best model
best_preds = results[best_name]['preds']
preds_df = test[['Date', 'Weekly_Sales']].copy()
preds_df['Pred'] = best_preds

# Aggregate by Date for clearer time series visualization
actual_agg = preds_df.groupby('Date')['Weekly_Sales'].sum().sort_index()
pred_agg = preds_df.groupby('Date')['Pred'].sum().sort_index()

plt.figure(figsize=(12,6))
plt.plot(actual_agg.index, actual_agg.values, label='Actual (aggregated)', linewidth=2)
plt.plot(pred_agg.index, pred_agg.values, label=f'Predicted ({best_name}, aggregated)', linewidth=2, alpha=0.8)
plt.legend()
plt.title('Aggregated Actual vs Predicted Weekly Sales (test set)')
plt.xlabel('Date')
plt.ylabel('Weekly Sales (aggregated across store-dept)')
plt.tight_layout()
plt.show()

# ---------------- OPTIONAL: Seasonal decomposition on aggregated national series ----------------
try:
    ts = sales.groupby('Date')['Weekly_Sales'].sum().sort_index()
    # Use resample to ensure consistent weekly frequency
    ts_weekly = ts.resample('W-MON').sum()
    decomp = seasonal_decompose(ts_weekly, period=52, model='additive', extrapolate_trend='freq')
    decomp.plot()
    plt.suptitle('Seasonal Decomposition (aggregated national sales)', fontsize=14)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f'Could not run seasonal decomposition: {e}')

# ---------------- OPTIONAL: save best model (pickle) ----------------
try:
    import joblib
    out_model_path = os.path.join('.', f'best_model_{best_name}.joblib')
    joblib.dump(results[best_name]['model'], out_model_path)
    print(f'Saved best model to: {out_model_path}')
except Exception:
    print('joblib not available or failed to save model — skipping save.')
