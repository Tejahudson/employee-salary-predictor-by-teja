import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Make sure this is imported
import joblib # To save and load the model

# --- 1. Load the Dataset ---
# IMPORTANT: Changed to 'ds_salaries.csv'. Make sure you have this file in your directory.
try:
    df = pd.read_csv('ds_salaries.csv')
    print("Dataset 'ds_salaries.csv' loaded successfully!")
    print(f"Dataset shape before handling NaNs: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'ds_salaries.csv' not found. Please make sure the CSV file is in the same directory as this script.")
    exit() # Exit if the file is not found

# --- 2. Handle Missing Values in Target Variable (if any, though ds_salaries is usually clean for salary) ---
# The target column in ds_salaries is 'salary_in_usd'
initial_rows = df.shape[0]
df.dropna(subset=['salary_in_usd'], inplace=True)
rows_after_dropping_salary_nan = df.shape[0]
if initial_rows - rows_after_dropping_salary_nan > 0:
    print(f"\nDropped {initial_rows - rows_after_dropping_salary_nan} rows with missing 'salary_in_usd' values.")
print(f"Dataset shape after handling 'salary_in_usd' NaNs: {df.shape}")


# --- 3. Exploratory Data Analysis (EDA) - Basic Checks ---
print("\n--- Dataset Info ---")
df.info()

print("\n--- Missing Values Check (after dropping Salary NaNs) ---")
print(df.isnull().sum())

# --- 4. Data Preprocessing ---

# Define features (X) and target (y) for the new dataset
# Adjust these column names to match 'ds_salaries.csv' exactly.
# We'll use 'experience_level' (categorical), 'job_title' (categorical),
# 'company_location' (categorical), 'remote_ratio' (numerical), and 'work_year' (numerical)
# as features. 'salary_in_usd' is the target.

features = ['experience_level', 'job_title', 'company_location', 'remote_ratio', 'work_year']
target = 'salary_in_usd'

# Check if all required columns exist in the DataFrame
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    print(f"\nError: The following required columns are missing from your dataset: {missing_cols}")
    print("Please adjust the 'features' and 'target' lists in the script to match your dataset's column names.")
    exit()

X = df[features]
y = df[target]

# Separate categorical and numerical features for preprocessing
numerical_features = ['remote_ratio', 'work_year'] # 'work_year' can be treated as numerical or categorical depending on model
categorical_features = ['experience_level', 'job_title', 'company_location']

# Handle potential NaNs in features (if any remain after dropping target NaNs)
# For numerical features, we'll impute with the mean
# For categorical features, we'll impute with the most frequent value (mode)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any
)

# --- 5. Train the Machine Learning Model ---

# Create a pipeline that first preprocesses the data and then trains the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42)) # Using RandomForestRegressor
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete!")

# --- 6. Evaluate the Model ---
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Evaluation on Test Set:")
print(f"R-squared (R2) Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

# --- 7. Save the Trained Model ---
model_filename = 'salary_prediction_model.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved successfully as '{model_filename}'")

print("\nNow, update your Streamlit app (app.py) to reflect the new input features!")
