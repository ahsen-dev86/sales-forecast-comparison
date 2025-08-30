# 📊 Sales Forecasting with Machine Learning

This project is part of my **Machine Learning Internship at Elevvo Pathways**.  
It demonstrates how machine learning can be applied to **retail sales forecasting** using historical Walmart-style data.  

We built a full pipeline that:
1. Loads and cleans sales, store, and external feature datasets
2. Engineers **time-series features** (lags, rolling windows, holiday indicators, etc.)
3. Trains and evaluates multiple models:
   - Linear Regression  
   - Random Forest  
   - XGBoost (if installed)  
   - LightGBM (if installed)
4. Visualizes results in an **interactive Streamlit dashboard**
5. Selects the best model based on **RMSE (Root Mean Squared Error)**

---

## 🚀 Key Findings
- Among the models tested, **Random Forest** achieved the best accuracy in predicting weekly sales.
- Feature engineering (lags & rolling averages) significantly improved performance.
- An interactive **Streamlit app** was built to allow non-technical users to explore results.

---

---

## ⚙️ Installation

Clone the repo:
```bash
git clone https://github.com/your-username/sales-forecasting.git
cd sales-forecasting
Create and activate a virtual environment:


python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\\Scripts\\activate    # On Windows
Install dependencies:


pip install -r requirements.txt
▶️ Usage
1. Run training script
bash
Copy code
python sales_forecasting.py
This will:

Load and merge the data

Create features

Train the models

Print RMSE scores

Save the best model as a .joblib file

2. Launch Streamlit dashboard
bash
Copy code
streamlit run app.py
This will start an interactive app where you can:

View the project overview

Compare models on separate tabs

See Actual vs. Predicted sales plots

Explore seasonal decomposition of sales trends

📊 Example Output
Best Model: Random Forest 🌲
RMSE: ~[Insert your RMSE value here]

The dashboard shows:

📈 Actual vs. Predicted weekly sales trends

🔎 Seasonal decomposition of sales patterns

📊 Per-model performance comparisons

🛠️ Requirements
Python 3.8+

pandas, numpy, matplotlib

scikit-learn

streamlit

statsmodels

joblib

(optional) xgboost, lightgbm

Install all with:

bash
Copy code
pip install -r requirements.txt
🙌 Acknowledgements
This project is part of the Machine Learning Internship program by Elevvo Pathways.
Dataset inspired by the Walmart Sales Forecasting dataset.



🏷️ Tags
#MachineLearning #DataScience #TimeSeries #Forecasting #RandomForest #Streamlit #Python
