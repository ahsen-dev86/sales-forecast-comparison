import os
import sys
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Optional imports
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
import joblib

warnings.filterwarnings("ignore")

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 Sales Forecasting Project")
st.markdown(
    """
    This project demonstrates a **retail sales forecasting pipeline**. 
    Using historical Walmart-style data (`train.csv`, `features.csv`, `stores.csv`), we:
    1. Load and merge datasets
    2. Engineer **time-series features** (lags, rolling averages, holidays, etc.)
    3. Train multiple ML models (**Linear Regression, Random Forest, XGBoost, LightGBM**)
    4. Evaluate them on test data using **RMSE**
    5. Visualize actual vs. predicted sales

    👉 Navigate through the tabs below to see model performances and plots.
    """
)

# ---------------- LOAD DATA ----------------
DATA_DIR = 'data'
SALES_CSV = os.path.join(DATA_DIR, 'train.csv')
FEATURES_CSV = os.path.join(DATA_DIR, 'features.csv')
STORES_CSV = os.path.join(DATA_DIR, 'stores.csv')

for fp in [SALES_CSV, FEATURES_CSV, STORES_CSV]:
    if not os.path.exists(fp):
        st.error(f"Required file not found: {fp}")
        st.stop()

sales = pd.read_csv(SALES_CSV, parse_dates=['Date'])
feat = pd.read_csv(FEATURES_CSV, parse_dates=['Date'])
stores = pd.read_csv(STORES_CSV)

if 'weekly_sales' in sales.columns and 'Weekly_Sales' not in sales.columns:
    sales.rename(columns={'weekly_sales': 'Weekly_Sales'}, inplace=True)

# Merge
df = sales.merge(feat, on=['Store', 'Date'], how='left').merge(stores, on='Store', how='left')
df.sort_values(['Store', 'Dept', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Fix IsHoliday
if 'IsHoliday_x' in df.columns and 'IsHoliday_y' in df.columns:
    df['IsHoliday'] = df[['IsHoliday_x', 'IsHoliday_y']].max(axis=1)
    df.drop(columns=['IsHoliday_x', 'IsHoliday_y'], inplace=True)
elif 'IsHoliday' not in df.columns:
    df['IsHoliday'] = 0
df['IsHoliday'] = df['IsHoliday'].astype(int)

# Fill missing externals
external_numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
for col in external_numeric_cols:
    if col not in df.columns:
        df[col] = np.nan
    med = df[col].median(skipna=True)
    if pd.isna(med):
        med = 0.0
    df[col].fillna(med, inplace=True)

# Time features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
df['dayofweek'] = df['Date'].dt.dayofweek
df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
df['quarter'] = df['Date'].dt.quarter

# Lag & rolling features
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

initial_rows = len(df)
df.dropna(inplace=True)
dropped = initial_rows - len(df)

# Feature list
base_time_feats = ['year', 'month', 'weekofyear', 'dayofweek', 'is_month_start', 'is_month_end', 'quarter', 'IsHoliday']
external_feats_present = [c for c in external_numeric_cols if c in df.columns]
lag_feats = [f'lag_{l}' for l in lags if f'lag_{l}' in df.columns]
rolling_feats = [f'rolling_mean_{w}' for w in rolling_windows if f'rolling_mean_{w}' in df.columns] + \
                [f'rolling_std_{w}' for w in rolling_windows if f'rolling_std_{w}' in df.columns]

features = base_time_feats + external_feats_present + lag_feats + rolling_feats

# Train/test split
cutoff_date = df['Date'].quantile(0.8)
train = df[df['Date'] <= cutoff_date].copy()
test = df[df['Date'] > cutoff_date].copy()

X_train, y_train = train[features], train['Weekly_Sales']
X_test, y_test = test[features], test['Weekly_Sales']

# ---------------- MODELS ----------------
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=0),
}
if XGB_AVAILABLE:
    models['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0)
if LGB_AVAILABLE:
    models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=0)

results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"rmse": rmse, "preds": preds, "model": model}
    except Exception as e:
        st.warning(f"Error training {name}: {e}")

if not results:
    st.error("No models trained successfully.")
    st.stop()

best_name = min(results.keys(), key=lambda k: results[k]['rmse'])
best_rmse = results[best_name]['rmse']

# ---------------- STREAMLIT TABS ----------------
tabs = st.tabs(list(results.keys()))

for i, (name, res) in enumerate(results.items()):
    with tabs[i]:
        st.subheader(f"{name} Results")
        st.write(f"**RMSE:** {res['rmse']:.4f}")

        # Plot actual vs predicted (aggregated)
        preds_df = test[['Date', 'Weekly_Sales']].copy()
        preds_df['Pred'] = res['preds']
        actual_agg = preds_df.groupby('Date')['Weekly_Sales'].sum().sort_index()
        pred_agg = preds_df.groupby('Date')['Pred'].sum().sort_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(actual_agg.index, actual_agg.values, label='Actual', linewidth=2)
        ax.plot(pred_agg.index, pred_agg.values, label='Predicted', linewidth=2, alpha=0.8)
        ax.legend()
        ax.set_title(f"Actual vs Predicted Sales ({name})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weekly Sales (aggregated)")
        st.pyplot(fig)

# ---------------- OPTIONAL: Seasonal decomposition ----------------
with st.expander("📈 Seasonal Decomposition (National Sales)"):
    try:
        ts = sales.groupby('Date')['Weekly_Sales'].sum().sort_index()
        ts_weekly = ts.resample('W-MON').sum()
        decomp = seasonal_decompose(ts_weekly, period=52, model='additive', extrapolate_trend='freq')
        fig = decomp.plot()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not run seasonal decomposition: {e}")

# ---------------- Save Best Model ----------------
try:
    out_model_path = os.path.join('.', f'best_model_{best_name}.joblib')
    joblib.dump(results[best_name]['model'], out_model_path)
    st.success(f"Best model saved as: {out_model_path}")
except Exception:
    st.warning("Could not save best model.")
