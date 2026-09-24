import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("house_price.csv")

# Input features
X = data[["area", "bedrooms", "bathrooms"]]

# Target value
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)

print("Model trained successfully!")
print("Model accuracy:", accuracy)

# Save model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as house_price_model.pkl")