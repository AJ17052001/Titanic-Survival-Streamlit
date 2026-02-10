import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Page Config
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title(" Titanic Survival Classification Dashboard")

# 1. Load and Preprocess Dataset
@st.cache_data
def load_data():
    # Attempt to load the local file
    try:
        df = pd.read_csv('Titanic-Dataset.csv')
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Titanic-Dataset.csv' is in the same directory.")
        return None

    # Cleaning
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = df.dropna(subset=['Embarked'])
    
    # Feature Engineering
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

df = load_data()

if df is not None:
    # 2. Features and Target
    X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
    y = df['Survived']

    # 3. Model Logic
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 4. Streamlit UI Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Evaluation")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc:.2%}")
        
        st.text("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    st.divider()
    
    # Optional: Data Preview
    if st.checkbox("Show Raw Processed Data"):
        st.write(df.head())
