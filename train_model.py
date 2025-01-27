import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("house_price_data_10000_rows.csv")

# Preprocess the data
lb = LabelEncoder()
n_df = df.drop(["Price (INR)"], axis="columns")
n_df['location_n'] = lb.fit_transform(n_df['Location'])
n_df['buildtype_n'] = lb.fit_transform(n_df['Building Type'])
n_df['Furnished_n'] = lb.fit_transform(n_df['Furnishing Status'])
n_df = n_df.drop(['Location', 'Building Type', 'Furnishing Status'], axis='columns')

# Train the model
model = LinearRegression()
model.fit(n_df, df[["Price (INR)"]])

# Save the model again
joblib.dump(model, "model.pkl")
print("Model retrained and saved as model.pkl")
