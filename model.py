import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
_df = pd.read_csv('loan_data.csv')

# Fill missing values
for column in _df.columns:
    if _df[column].dtype == 'object':
        _df[column].fillna(_df[column].mode()[0], inplace=True)
    else:
        _df[column].fillna(_df[column].median(), inplace=True)

# Encode categorical columns
label_encoders = {}
for column in _df.select_dtypes(include=['object']).columns:
    if column != 'Loan_Status':
        le = LabelEncoder()
        _df[column] = le.fit_transform(_df[column])
        label_encoders[column] = le

# Encode target
_df['Loan_Status'] = _df['Loan_Status'].map({'Y': 1, 'N': 0})

X = _df.drop(['Loan_ID', 'Loan_Status'], axis=1, errors='ignore')
y = _df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'loan_prediction_model.pkl')
print('Model trained and saved successfully!')