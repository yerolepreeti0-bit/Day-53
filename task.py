import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create Sample Dataset

data = pd.DataFrame({
    'age': [25, 32, 47, 51, 62, 23, 36, 44],
    'salary': [50000, 60000, 80000, 90000, 120000, 45000, 70000, 85000],
    'city': ['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Delhi', 'Mumbai', 'Bangalore', 'Chennai'],
    'purchased': [0, 1, 0, 1, 1, 0, 1, 0]
})

# Features and target
X = data[['age', 'salary', 'city']]
y = data['purchased']


# 2. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define Column Types

numeric_columns = ['age', 'salary']
categorical_columns = ['city']


# 4. Preprocessing (ColumnTransformer)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_columns),
    ('cat', OneHotEncoder(), categorical_columns)
])


# 5. Create Pipeline

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LogisticRegression())
])


# 6. Train Model

pipeline.fit(X_train, y_train)


# 7. Predict
y_pred = pipeline.predict(X_test)

# 8. Evaluate
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))