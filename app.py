import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set Streamlit page configuration (this must be the first Streamlit command)
st.set_page_config(page_title="Iris Classification with k-NN", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .section-header {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        text-align: center;
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

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    start_time = time.time()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    return model, scaler, X_test_scaled, y_test, training_time

model, scaler, X_test_scaled, y_test, training_time = train_model(X, y)

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Iris Classification with k-NN</h1>", unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.markdown("<div class='section-header'>Input Features</div>", unsafe_allow_html=True)
    st.markdown("Adjust the sliders to input iris measurements:")
    
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.7, 0.1)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.5, 0.1)
    
    predict_button = st.button('Predict Iris Species')

# Main Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='section-header'>Model Prediction</div>", unsafe_allow_html=True)
    
    if predict_button:
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)
        
        iris_species = iris.target_names[prediction[0]]
        st.markdown(f"<h3>Predicted Species: <span style='color: #FF5733;'>{iris_species}</span></h3>", unsafe_allow_html=True)
        
        st.markdown("### Prediction Probability")
        prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
        st.dataframe(prob_df.style.background_gradient(cmap='Blues').format("{:.2%}"))
        
        # Nearest Neighbors
        st.markdown("### Nearest Neighbors")
        distances, indices = model.kneighbors(user_input_scaled)
        neighbors_df = pd.DataFrame({
            'Neighbor': range(1, 6),
            'Distance': distances[0],
            'Index in Training Set': indices[0]
        })
        st.dataframe(neighbors_df.style.background_gradient(cmap='Greens'))

with col2:
    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time": training_time
    }
    
    for metric, value in metrics.items():
        st.markdown(f"""
        <div class='metric-card'>
            <h4>{metric}</h4>
            <h2 style='color: #4CAF50;'>{value:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

# Feature Importance Section
st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)

feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': perm_importance.importances_mean
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis", ax=ax)
ax.set_title("Feature Importance (Permutation Importance)")
ax.set_xlabel("Mean Importance Score")
ax.set_ylabel("Feature")
st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Developed using Streamlit and scikit-learn</p>", unsafe_allow_html=True)
