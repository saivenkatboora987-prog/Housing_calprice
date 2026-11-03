# California House Price Predictor

A machine learning project that predicts median house value in California districts using Linear Regression, with a web interface for live predictions.

## Problem
Estimate the median house value of a California district from census features such as income, house age, and location.

## Dataset
California Housing dataset (scikit-learn / StatLib, 1990 US census): 20,640 block groups, 8 numeric features, no missing values. Target: `MedHouseVal` (in $100,000s).

## Approach
1. **EDA:** correlation heatmap and scatter plots to inspect relationships
2. **Multicollinearity check:** Variance Inflation Factor (VIF); removed `AveBedrms` and `Longitude`
3. **Features used (6):** MedInc, HouseAge, AveRooms, Population, AveOccup, Latitude
4. **Train/test split:** 80/20 (16,512 / 4,128 samples)
5. **Model:** Linear Regression (scikit-learn); verified predictions manually using the intercept and coefficients
6. **Deployment:** model saved with joblib and served through a web app [Flask - verify]

## Results
<img width="560" height="178" alt="image" src="https://github.com/user-attachments/assets/fcb602cc-efbb-484b-b723-c0cb3305c266" />


## How to run
```bash
git clone https://github.com/saivenkatboora987-prog/Housing_calprice.git
cd Housing_calprice
pip install flask scikit-learn pandas numpy joblib
python app.py
```

## Tech stack
Python, Pandas, NumPy, scikit-learn, statsmodels, Matplotlib, Seaborn, joblib [, Flask]

## Future improvements
- Compare with Random Forest and Gradient Boosting
- Add feature scaling and cross-validation
- Restore location information in a better form (for example, distance to major cities)
