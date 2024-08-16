import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="🌸 Iris Classifier", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stApp { background-color: #f0f2f6; }
    .stButton>button { 
        color: #ffffff; 
        background-color: #4CAF50; 
        border-radius: 5px;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .section-header {
        background-color: #4CAF50;
        color: white;
        padding: 0.7rem;
        border-radius: 5px;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare the data
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris

X, y, iris = load_data()

# Define models with emojis
models = {
    "🌳 Decision Tree": DecisionTreeClassifier(),
    "🌲 Random Forest": RandomForestClassifier(),
    "🏘️ K-Nearest Neighbors": KNeighborsClassifier(),
    "🧠 Support Vector Machine": SVC(probability=True)
}

# Train the selected model
@st.cache_resource
def train_model(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = models[model_name]
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #4CAF50; margin-bottom: 2rem;'>🌸 Iris Classification App</h1>", unsafe_allow_html=True)

# App description
st.markdown("""
This app uses machine learning to classify Iris flowers based on their sepal and petal measurements.
Choose a model, input the flower's measurements, and let the AI predict the Iris species! 🌺
""")

# Sidebar for user input and model selection
with st.sidebar:
    st.markdown("<div class='section-header'>🤖 Model Selection</div>", unsafe_allow_html=True)
    selected_model = st.selectbox("Choose a model", list(models.keys()))
    
    st.markdown("<div class='section-header'>📏 Input Features</div>", unsafe_allow_html=True)
    st.markdown("Adjust the sliders to input iris measurements:")
    
    sepal_length = st.slider('🔹 Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.slider('🔸 Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
    petal_length = st.slider('🔹 Petal Length (cm)', 1.0, 7.0, 4.7, 0.1)
    petal_width = st.slider('🔸 Petal Width (cm)', 0.1, 2.5, 1.5, 0.1)
    
    predict_button = st.button('🔮 Predict Iris Species')

# Train the selected model
model, scaler, X_test_scaled, y_test = train_model(X, y, selected_model)

# Main Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='section-header'>🔍 Model Prediction</div>", unsafe_allow_html=True)
    
    if predict_button:
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)
        
        iris_species = iris.target_names[prediction[0]]
        st.markdown(f"<h3>Predicted Species: <span style='color: #FF5733;'>{iris_species}</span> 🌼</h3>", unsafe_allow_html=True)
        
        st.markdown("### Prediction Probability")
        prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
        st.dataframe(prob_df.style.background_gradient(cmap='Greens').format("{:.2%}"))

with col2:
    st.markdown("<div class='section-header'>📊 Model Performance</div>", unsafe_allow_html=True)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    col_acc, col_prec = st.columns(2)
    col_rec, col_f1 = st.columns(2)
    
    with col_acc:
        st.markdown("<div class='metric-card'><div class='metric-value'>{:.2f}%</div><div class='metric-label'>Accuracy 🎯</div></div>".format(accuracy*100), unsafe_allow_html=True)
    with col_prec:
        st.markdown("<div class='metric-card'><div class='metric-value'>{:.2f}%</div><div class='metric-label'>Precision ✅</div></div>".format(precision*100), unsafe_allow_html=True)
    with col_rec:
        st.markdown("<div class='metric-card'><div class='metric-value'>{:.2f}%</div><div class='metric-label'>Recall 🔍</div></div>".format(recall*100), unsafe_allow_html=True)
    with col_f1:
        st.markdown("<div class='metric-card'><div class='metric-value'>{:.2f}%</div><div class='metric-label'>F1 Score 🏆</div></div>".format(f1*100), unsafe_allow_html=True)

# Feature Importance Section
st.markdown("<div class='section-header'>🔑 Feature Importance</div>", unsafe_allow_html=True)

# Calculate feature importance
perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(feature_importance['Feature'], feature_importance['Importance'], color='#4CAF50')
ax.set_title(f"Feature Importance for {selected_model}", fontsize=16)
ax.set_xlabel("Features", fontsize=12)
ax.set_ylabel("Importance Score", fontsize=12)
plt.xticks(rotation=45, ha='right')

# Add value labels on top of each bar
for i, v in enumerate(feature_importance['Importance']):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
The feature importance chart shows how much each feature contributes to the model's predictions.
Higher scores indicate more important features. This can help understand which measurements are most crucial for identifying Iris species.
""")

# Model Description
st.markdown("<div class='section-header'>📘 Model Information</div>", unsafe_allow_html=True)
model_descriptions = {
    "🌳 Decision Tree": "A tree-like model that makes decisions based on feature thresholds.",
    "🌲 Random Forest": "An ensemble of decision trees for improved accuracy and robustness.",
    "🏘️ K-Nearest Neighbors": "Classifies based on the majority class of K nearest data points.",
    "🧠 Support Vector Machine": "Finds the optimal hyperplane to separate classes in high-dimensional space."
}
st.write(f"**{selected_model}**: {model_descriptions[selected_model]}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Developed with ❤️ using Streamlit and scikit-learn</p>
    <p>🌸 Iris Classification App | © 2024 All rights reserved</p>
</div>
""", unsafe_allow_html=True)