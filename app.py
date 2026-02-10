import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Page configuration
st.set_page_config(page_title="Titanic Survival Dashboard", layout="wide")

st.title(" Titanic Survival Prediction")
st.markdown("This app uses Logistic Regression to predict passenger survival based on the Titanic dataset.")

# 1. Load Data
@st.cache_data
def get_clean_data():
    # Make sure the filename matches exactly what you uploaded to GitHub
    try:
        df = pd.read_csv('Titanic-Dataset.csv')
    except FileNotFoundError:
        st.error(" 'Titanic-Dataset.csv' not found. Ensure it is in the same GitHub folder as this script.")
        return None

    # Preprocessing (matching your original logic)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df = df.dropna(subset=['Embarked'])
    
    # Feature Engineering
    df_clean = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df_clean

df = get_clean_data()

if df is not None:
    # 2. Define Features and Target
    # Based on your image, these are the columns used:
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    X = df[features]
    y = df['Survived']

    # 3. Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 4. Display Metrics and Visuals
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader(" Performance Metrics")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc:.2%}")
        
        st.write("**Classification Report:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().style.highlight_max(axis=0))

    with col2:
        st.subheader(" Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    st.divider()

    # 5. Interactive Prediction Section
    st.subheader(" Predict Survival")
    st.write("Enter passenger details below to see if they would have survived:")
    
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        p_class = st.selectbox("Ticket Class (1st, 2nd, 3rd)", [1, 2, 3])
        age = st.slider("Age", 0, 100, 30)
    with p_col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        fare = st.number_input("Fare Paid", value=32.0)
    with p_col3:
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

    # Convert inputs to model format
    sex_male = 1 if sex == "Male" else 0
    # Assuming standard Southampton embarkation for prediction
    input_data = [[p_class, age, sibsp, parch, fare, sex_male, 0, 1]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if st.button("Predict"):
        if prediction[0] == 1:
            st.success("The model predicts this passenger would have **SURVIVED**.")
        else:
            st.error("The model predicts this passenger would **NOT HAVE SURVIVED**.")
