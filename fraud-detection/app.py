import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache_resource
def train_model(csv_path="creditcard.csv"):
    """Train a Logistic Regression model on a balanced sample of the dataset."""
    data = pd.read_csv(csv_path)

    # Separate legitimate and fraud transactions
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    # Undersample legitimate transactions so both classes have the same size
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    # Split into training and testing data
    X = data.drop(columns="Class")
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.shape[1], train_acc, test_acc


st.title("Credit Card Fraud Detection Model")

try:
    model, n_features, train_acc, test_acc = train_model()
except FileNotFoundError:
    st.error("creditcard.csv not found. Download it from Kaggle and place it next to app.py.")
    st.stop()

st.sidebar.header("Model accuracy")
st.sidebar.write(f"Training accuracy: {train_acc:.2%}")
st.sidebar.write(f"Test accuracy: {test_acc:.2%}")

st.write(f"Enter all {n_features} feature values separated by commas "
         "(Time, V1 to V28, Amount). Example rows are in sample_inputs.txt.")
input_text = st.text_input("Transaction features")

if st.button("Submit"):
    try:
        values = [float(v) for v in input_text.split(",")]
        if len(values) != n_features:
            raise ValueError
    except ValueError:
        st.error(f"Please enter exactly {n_features} numbers separated by commas.")
    else:
        features = np.asarray(values, dtype=np.float64).reshape(1, -1)
        prediction = model.predict(features)[0]
        st.write("Fraud Transaction" if prediction == 1 else "Legitimate Transaction")
