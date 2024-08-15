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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set page configuration
st.set_page_config(page_title="Iris Classification App", layout="wide", initial_sidebar_state="expanded")

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
    }
    .stButton>button:hover { background-color: #45a049; }
    .section-header {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    .stSelectbox { margin-bottom: 1rem; }
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

# Define models
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train the selected model
@st.cache_resource(ttl=600)  # Cache for 10 minutes
def train_model(X, y, model_name):
    X += np.random.normal(0, 0.1, X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = models[model_name]
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    return model, scaler, X_test_scaled, y_test, training_time

# Streamlit app title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>ML Classification with Iris Data</h1>", unsafe_allow_html=True)

# Sidebar for user input and model selection
with st.sidebar:
    st.markdown("<div class='section-header'>Model Selection</div>", unsafe_allow_html=True)
    selected_model = st.selectbox("Choose a model", list(models.keys()))
    
    st.markdown("<div class='section-header'>Input Features</div>", unsafe_allow_html=True)
    st.markdown("Adjust the sliders to input iris measurements:")
    
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.7, 0.1)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.5, 0.1)
    
    predict_button = st.button('Predict Iris Species')

# Train the selected model
model, scaler, X_test_scaled, y_test, training_time = train_model(X, y, selected_model)

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
        
        if selected_model == "K-Nearest Neighbors":
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
    
    # Display metrics inline using columns for a tighter layout
    col_acc, col_prec, col_rec, col_f1, col_time = st.columns(5)
    
    col_acc.metric("Accuracy", f"{accuracy:.4f}")
    col_prec.metric("Precision", f"{precision:.4f}")
    col_rec.metric("Recall", f"{recall:.4f}")
    col_f1.metric("F1 Score", f"{f1:.4f}")
    col_time.metric("Training Time", f"{training_time:.4f}s")

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
ax.set_title(f"Feature Importance for {selected_model}")
ax.set_xlabel("Mean Importance Score")
ax.set_ylabel("Feature")
st.pyplot(fig)

# Model Comparison Section
st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)

@st.cache_data
def compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": accuracy})
    
    return pd.DataFrame(results)

# Interactive Data Exploration
st.markdown("<div class='section-header'>Interactive Data Exploration</div>", unsafe_allow_html=True)

df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

feature_x = st.selectbox('Select feature for x-axis', iris.feature_names)
feature_y = st.selectbox('Select feature for y-axis', iris.feature_names)

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df[feature_x], df[feature_y], c=df['target'], cmap='viridis')
ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)
ax.set_title(f'{feature_y} vs {feature_x}')
plt.colorbar(scatter)
st.pyplot(fig)

# Confusion Matrix Visualization
st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)

y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix for {selected_model}')
st.pyplot(fig)

# Hyperparameter Tuning
st.markdown("<div class='section-header'>Hyperparameter Tuning</div>", unsafe_allow_html=True)

st.write("Adjust hyperparameters and see how they affect model performance:")

if selected_model == "K-Nearest Neighbors":
    n_neighbors = st.slider('Number of neighbors', 1, 20, 5)
    models["K-Nearest Neighbors"] = KNeighborsClassifier(n_neighbors=n_neighbors)
elif selected_model == "Support Vector Machine":
    C = st.slider('C (Regularization parameter)', 0.01, 10.0, 1.0)
    kernel = st.selectbox('Kernel', ['rbf', 'linear', 'poly'])
    models["Support Vector Machine"] = SVC(C=C, kernel=kernel, probability=True)
elif selected_model == "Decision Tree":
    max_depth = st.slider('Max depth', 1, 20, 5)
    models["Decision Tree"] = DecisionTreeClassifier(max_depth=max_depth)
elif selected_model == "Random Forest":
    n_estimators = st.slider('Number of trees', 1, 200, 100)
    max_depth = st.slider('Max depth', 1, 20, 5)
    models["Random Forest"] = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

if st.button('Retrain Model'):
    model, scaler, X_test_scaled, y_test, training_time = train_model(X, y, selected_model)
    st.success('Model retrained successfully!')

    # Update performance metrics
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    col_acc, col_prec, col_rec, col_f1, col_time = st.columns(5)
    
    col_acc.metric("Accuracy", f"{accuracy:.4f}")
    col_prec.metric("Precision", f"{precision:.4f}")
    col_rec.metric("Recall", f"{recall:.4f}")
    col_f1.metric("F1 Score", f"{f1:.4f}")
    col_time.metric("Training Time", f"{training_time:.4f}s")


comparison_df = compare_models(X, y)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Model", data=comparison_df, palette="viridis", ax=ax)
ax.set_title("Model Comparison")
ax.set_xlabel("Accuracy")
st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Developed with ❤️ using Streamlit and scikit-learn</p>", unsafe_allow_html=True)
