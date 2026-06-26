import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# 1. LOAD
df = pd.read_csv('data/train.csv')

# 2. FEATURES — these columns exist in Kaggle CSV
FEATURES = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'GarageCars', 'OverallQual']
TARGET = 'SalePrice'

# Drop rows with missing values in selected columns
df = df[FEATURES + [TARGET]].dropna()

X = df[FEATURES]
y = df[TARGET]

# 3. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN 3 MODELS
models = {
    'Linear Regression': LinearRegression(),
    'Ridge':             Ridge(alpha=1.0),
    'Lasso':             Lasso(alpha=100),
}

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n{name}")
    print(f"  R²  : {r2_score(y_test, y_pred):.4f}")
    print(f"  MAE : ${mean_absolute_error(y_test, y_pred):,.0f}")
    print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
    print(f"  CV R²: {cross_val_score(pipe, X, y, cv=5, scoring='r2').mean():.4f}")

    joblib.dump(pipe, f"models/{name.replace(' ', '_').lower()}.pkl")

# 5. PLOTS
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()
print("\nDone. Models saved to models/")