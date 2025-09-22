import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# ----------------------------
# 1. Load dataset
# ----------------------------
data = pd.read_csv(r"data\student_performance_prediction.csv")
data.columns = data.columns.str.strip()

# ----------------------------
# 2. Drop completely useless columns
# ----------------------------
if 'Student ID' in data.columns:
    data = data.drop(columns=['Student ID'])

# ----------------------------
# 3. Drop rows with missing target
# ----------------------------
data = data.dropna(subset=['Passed'])

# ----------------------------
# 4. Fix negative/invalid numeric values
# ----------------------------
median_hours = data.loc[data['Study Hours per Week'] >= 0, 'Study Hours per Week'].median()
data.loc[data['Study Hours per Week'] < 0, 'Study Hours per Week'] = median_hours
data.loc[data['Attendance Rate'] > 100, 'Attendance Rate'] = 100
data.loc[data['Previous Grades'] > 100, 'Previous Grades'] = 100

# ----------------------------
# 5. Additional features
# ----------------------------
data['Study Efficiency'] = data['Previous Grades'] / (data['Study Hours per Week'] + 1)
data['Academic Engagement'] = (data['Attendance Rate'] * data['Study Hours per Week']) / 100

# Categorize study hours
def categorize_study_hours(hours):
    if hours < 10:
        return 'Low'
    elif hours < 20:
        return 'Medium'
    else:
        return 'High'

def categorize_attendance(rate):
    if rate < 70:
        return 'Poor'
    elif rate < 85:
        return 'Average'
    else:
        return 'Good'

data['Study Hours Category'] = data['Study Hours per Week'].apply(categorize_study_hours)
data['Attendance Category'] = data['Attendance Rate'].apply(categorize_attendance)

# ----------------------------
# 6. Collapse target to binary
# ----------------------------
# 0 = Fail, 1 = Pass (combine classes 1 & 2)
data['Passed_binary'] = data['Passed'].replace({0: 0, 1: 1, 2: 1})

# ----------------------------
# 7. Impute remaining missing values
# ----------------------------
numeric_cols = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades',
                'Study Efficiency', 'Academic Engagement']
cat_cols = ['Participation in Extracurricular Activities', 'Parent Education Level',
            'Study Hours Category', 'Attendance Category']

# Numeric: median
imputer_num = SimpleImputer(strategy='median')
data[numeric_cols] = imputer_num.fit_transform(data[numeric_cols])

# Categorical: most frequent
imputer_cat = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])

# ----------------------------
# 8. Features and target
# ----------------------------
X = data[numeric_cols + cat_cols]
y = data['Passed_binary']

# ----------------------------
# 9. Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# 10. Preprocessing pipelines
# ----------------------------
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, cat_cols)
])

# ----------------------------
# 11. Random Forest Pipeline
# ----------------------------
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'))
])

# ----------------------------
# 12. Train model
# ----------------------------
rf_model.fit(X_train, y_train)

# ----------------------------
# 13. Save model
# ----------------------------
import os
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(rf_model, r"model\rf_binary_model.pkl")
print("Random Forest model saved as 'model\\rf_binary_model.pkl'")

# ----------------------------
# 14. Evaluate model
# ----------------------------
y_pred = rf_model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.4f}")
