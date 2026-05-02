import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Set page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')
    return df

# Cache model training
@st.cache_resource
def train_model(df_clean):
    X = df_clean.drop('survived', axis=1)
    y = df_clean['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return rf, acc, X.columns.tolist()

def preprocess_data(df):
    # Engineering
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Preprocessing
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['fare'] = df['fare'].fillna(df['fare'].median())
    
    # Drops
    cols_to_drop = ['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone']
    df_clean = df.drop(cols_to_drop, axis=1)
    
    # Encoding
    le = LabelEncoder()
    df_clean['sex'] = le.fit_transform(df_clean['sex'])
    df_clean['embarked'] = le.fit_transform(df_clean['embarked'])
    
    return df_clean

# Main App Logic
st.title("🚢 Titanic Survival Prediction Dashboard")
st.markdown("---")

df = load_data()
df_processed = preprocess_data(df.copy())
model, accuracy, feature_cols = train_model(df_processed)

# Sidebar - Passenger Input
st.sidebar.header("Input Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3], index=2)
sex = st.sidebar.selectbox("Gender", ["male", "female"], index=0)
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("Fare Paid", 0.0, 600.0, 7.25)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0)

# Calculate derived features for prediction
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"S": 2, "C": 0, "Q": 1}
embarked_encoded = embarked_map[embarked]

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📊 EDA Dashboard", "🧠 Model Performance", "🔮 Predict Survival"])

with tab1:
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Survival Rate by Gender")
        fig1, ax1 = plt.subplots()
        sns.barplot(x='sex', y='survived', data=df, ax=ax1, palette='viridis')
        st.pyplot(fig1)
        
    with col2:
        st.write("#### Survival Rate by Class")
        fig2, ax2 = plt.subplots()
        sns.barplot(x='class', y='survived', data=df, ax=ax2, palette='magma')
        st.pyplot(fig2)
        
    st.write("#### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.subheader("Model Information (Random Forest)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
        
    with col_b:
        st.write("#### Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax_imp)
        st.pyplot(fig_imp)

with tab3:
    st.subheader("Survival Prediction")
    
    # Prediction logic
    input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded, family_size, is_alone]], 
                              columns=feature_cols)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.success("### Status: Likely to SURVIVE! ✅")
        else:
            st.error("### Status: Likely NOT to Survive ❌")
            
    with res_col2:
        st.write(f"#### Probability of Survival: `{probability*100:.2f}%`")
        st.progress(probability)

    st.info(f"Summary: Passenger is a {age} year old {sex} in class {pclass}, embarking from {embarked}.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Antigravity AI")
