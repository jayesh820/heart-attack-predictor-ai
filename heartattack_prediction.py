import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------ Load Data ------------------------
def load_data(path='heartattack_data.csv'):
    df = pd.read_csv(path)
    return df

# ------------------------ Train Model ------------------------
def train_model():
    df = load_data()
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'heart_model.pkl')
    print("✅ Model saved as 'heart_model.pkl'")

    # Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df.sort_values(by='Importance', ascending=False, inplace=True)

    print("\n📌 Feature Importance (Weightage):")
    print(feat_df)

    # Save Feature Importance Graph
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df)
    plt.title('Heart Attack Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\n🖼️ Feature importance graph saved as 'feature_importance.png'")

# ------------------------ Predict Using Input ------------------------
def predict_input():
    model = joblib.load('heart_model.pkl')

    print("\n🧾 Enter patient details to predict heart attack risk:")
    age = int(input("Age: "))
    sex = int(input("Sex (1=Male, 0=Female): "))
    cp = int(input("Chest Pain Type (0–3): "))
    trestbps = int(input("Resting Blood Pressure: "))
    chol = int(input("Cholesterol: "))
    fbs = int(input("Fasting Blood Sugar >120? (1/0): "))
    restecg = int(input("Rest ECG (0–2): "))
    thalach = int(input("Max Heart Rate: "))
    exang = int(input("Exercise Induced Angina (1/0): "))
    oldpeak = float(input("Oldpeak (ST depression): "))
    slope = int(input("Slope of ST Segment (0–2): "))
    ca = int(input("Number of Major Vessels (0–4): "))
    thal = int(input("Thal (0=Normal, 1=Fixed Defect, 2=Reversible): "))

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    print("\n🔎 Prediction Result:")
    if prediction == 1:
        print(f"⚠️ High Risk of Heart Attack ({proba*100:.2f}% confidence)")
    else:
        print(f"✅ Low Risk of Heart Attack ({(1 - proba)*100:.2f}% confidence)")

# ------------------------ Main Interface ------------------------
if __name__ == "__main__":
    print("🔬 Heart Attack Predictor - AI Based")
    print("Options:\n1. Train Model\n2. Predict Risk\n")
    choice = input("Choose (1/2): ")

    if choice == '1':
        train_model()
    elif choice == '2':
        predict_input()
    else:
        print("❌ Invalid option selected.")
