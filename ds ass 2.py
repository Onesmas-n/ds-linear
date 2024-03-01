import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Load the dataset
data = pd.read_csv("accident_data.csv")

# Step 2: Define dependent and independent variables
X = data[['road_conditions', 'weather', 'time_of_day']]
y = data['accident_severity']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Save the model for future use
joblib.dump(model, "accident_severity_model.pkl")

# Step 6: Use the model to predict accident severity for a hypothetical set of independent variables
# Example hypothetical data
hypothetical_data = [[2, 1, 3]]  # Road conditions: Moderate, Weather: Clear, Time of day: Evening

# Load the model
loaded_model = joblib.load("accident_severity_model.pkl")

# Predict accident severity
predicted_severity = loaded_model.predict(hypothetical_data)
print("Predicted accident severity:", predicted_severity)
