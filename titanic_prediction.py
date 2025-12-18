import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv('Titanic-Dataset.csv')

# Preprocessing
# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = df[features + ['Survived']]

# Encode categorical variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'].astype(str))
df['Embarked'] = df['Embarked'].fillna('Unknown')
df['Embarked'] = le_embarked.fit_transform(df['Embarked'].astype(str))

# Impute missing values for numeric features
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Fare'] = imputer.fit_transform(df[['Fare']])

# Split features and target
X = df[features]
y = df['Survived']

# Train/test split (for full prediction, train on all)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities
df['Survival_Probability'] = model.predict_proba(X)[:,1]

# Mark possibility
df['Prediction'] = df['Survival_Probability'].apply(
    lambda x: 'More possibility' if x > 0.5 else 'Less possibility'
)

# Show top/bottom 10 as example
print(df[['Survival_Probability', 'Prediction']].head(10))
print(df[['Survival_Probability', 'Prediction']].tail(10))

# Optionally, save predictions to file
df[['Survival_Probability', 'Prediction']].to_csv('Titanic_Predictions.csv', index=False)
