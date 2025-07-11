
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
X = df
y = data.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# App title
st.title("🔬 Breast Cancer Detection App")
st.markdown("Predict whether a tumor is **benign** or **malignant** based on input features.")

# Sidebar input
st.sidebar.header("Input Features")

def user_input_features():
    input_data = {}
    for feature in data.feature_names:
        input_data[feature] = st.sidebar.slider(
            label=feature,
            min_value=float(X[feature].min()),
            max_value=float(X[feature].max()),
            value=float(X[feature].mean())
        )
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Show user input
st.subheader("🧾 Entered Input:")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

# Show prediction
st.subheader("🔍 Prediction:")
st.write("**Malignant**" if prediction == 0 else "**Benign**")

st.subheader("📊 Prediction Probability:")
st.write(f"Malignant: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Benign: {prediction_proba[0][1]*100:.2f}%")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")

