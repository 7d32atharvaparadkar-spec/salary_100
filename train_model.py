import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("salary_100.csv")

# Convert text → numeric
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("salary_model.pkl","wb"))

print("✅ Model Trained")