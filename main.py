# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.max_columns', None)

df = pd.read_excel('Insurance and Medical Costs Data.xlsx')

df.head(20)

# %%
#B. Data Cleaning
# Get basic information about the dataset
df.info()

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

df = df.drop_duplicates()
# Check for duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())


# %%
#B Encoding Variables
# Convert categorical variables into numerical format
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # Mapping male to 0, female to 1
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})  # Mapping smoker 'yes' to 1, 'no' to 0

# One-hot encoding for the 'region' column
df = pd.get_dummies(df, columns=['region'], drop_first=True)  # Drop first category to avoid multicollinearity

# Display updated dataframe
df.head()


# %%
#C Data Extraction and Preperation
# Function to remove outliers using Interquartile Range (IQR)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from 'bmi' and 'charges' columns
df_cleaned = remove_outliers(df, 'bmi')
df_cleaned = remove_outliers(df_cleaned, 'charges')

# Apply log transformation to 'charges' to reduce skewness
df_cleaned['charges'] = np.log(df_cleaned['charges'])

# Apply Min-Max Normalization to 'bmi' for better scaling
df_cleaned['bmi'] = (df_cleaned['bmi'] - df_cleaned['bmi'].min()) / (df_cleaned['bmi'].max() - df_cleaned['bmi'].min())

# Display the shape of dataset before and after outlier removal
print("Original dataset shape:", df.shape)
print("Dataset shape after outlier removal:", df_cleaned.shape)

# Plot boxplots before transformation
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['age', 'bmi', 'children', 'charges']])
plt.title("Boxplots for Numerical Variables Before Transformation")
plt.show()

# Plot boxplots after log and normalization transformations
plt.figure(figsize=(12,6))
sns.boxplot(data=df_cleaned[['age', 'bmi', 'children', 'charges']])
plt.title("Boxplots for Numerical Variables After Log and Normalization Transformations")
plt.show()

# Function to detect outliers using IQR method
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers[[column]]

# Identify outliers in 'charges' and 'bmi' after transformations
charges_outliers = detect_outliers(df_cleaned, 'charges')
bmi_outliers = detect_outliers(df_cleaned, 'bmi')

# Display first 10 outlier values
print("\nOutliers in Charges after transformation:\n", charges_outliers.head(10))
print("\nOutliers in BMI after transformation:\n", bmi_outliers.head(10))


# %%
# Display the first 10 rows after encoding categorical variables
print("First 10 Rows of the Encoded Data:")
print(df.head(10))


# %%
#D. Data Analysis - model training & evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Prepare features and target from the cleaned data
X = df_cleaned.drop('charges', axis=1)
y = df_cleaned['charges']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

# 1. Linear Regression
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results["Linear Regression"] = {"MSE": mean_squared_error(y_test, y_pred_lr), "R2": r2_score(y_test, y_pred_lr)}

# 2. Tuned Gradient Boosting via GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_gbr = grid_search.best_estimator_
y_pred_best = best_gbr.predict(X_test)
results["Tuned Gradient Boosting"] = {"MSE": mean_squared_error(y_test, y_pred_best), "R2": r2_score(y_test, y_pred_best)}

# 3. Polynomial Regression (Degree 2)
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])
poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)
results["Polynomial Regression"] = {"MSE": mean_squared_error(y_test, y_pred_poly), "R2": r2_score(y_test, y_pred_poly)}

# 4. Stacking Regressor (Combining LR, Polynomial, and Tuned GBR)
estimators = [
    ('lr', LinearRegression()),
    ('poly', Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear', LinearRegression())
    ])),
    ('gbr', best_gbr)
]
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)
stacking_regressor.fit(X_train, y_train)
y_pred_stack = stacking_regressor.predict(X_test)
results["Stacking Regressor"] = {"MSE": mean_squared_error(y_test, y_pred_stack), "R2": r2_score(y_test, y_pred_stack)}

# Print a summary of the results
for model, metrics in results.items():
    print(f"{model}: MSE = {metrics['MSE']:.4f}, R² = {metrics['R2']:.4f}")


# %%
#D. Feature Importance Analysis
import matplotlib.pyplot as plt

# Extract feature importance from the tuned Gradient Boosting model
feature_importance = best_gbr.feature_importances_
features = X.columns

# Sort feature importance in descending order
sorted_idx = np.argsort(feature_importance)[::-1]

# Plot feature importance
plt.figure(figsize=(10,6))
plt.bar(range(len(features)), feature_importance[sorted_idx], align='center')
plt.xticks(range(len(features)), np.array(features)[sorted_idx], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importance from Tuned Gradient Boosting Model")
plt.show()


# %%
#D Residual Distribution Plot

# Calculate residuals for the best model (Stacking Regressor)
residuals = y_test - y_pred_stack

# Plot residuals
plt.figure(figsize=(10,6))
sns.histplot(residuals, kde=True, bins=30, color="royalblue")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution of Stacking Regressor Model")
plt.show()

#D Residuals vs. Predicted Values Scatter Plot
# Scatter plot of residuals vs. predicted values
plt.figure(figsize=(10,6))
plt.scatter(y_pred_stack, residuals, alpha=0.5, color="royalblue")
plt.axhline(y=0, color='red', linestyle='--')  # Reference line at 0 residual
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Charges (Stacking Regressor)")
plt.show()




