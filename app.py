import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fruit KNN Classifier",
    page_icon="",
    layout="wide"
)

st.title("Fruit Classification using KNN")

st.write(
    "Classify fruits using weight, diameter, and color "
    "with the K-Nearest Neighbors algorithm."
)

# ---------------------------------------------------------
# 1. Get the Data
# ---------------------------------------------------------

st.header("1. Get the Data and Describe")

uploaded_file = st.file_uploader(
    "Upload your fruit dataset file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to start the project.")
    st.stop()

# Support CSV and Excel files

if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("Dataset")
st.dataframe(df, use_container_width=True)

st.subheader("First Five Records")
st.dataframe(df.head())

st.subheader("Dataset Description")
st.dataframe(df.describe(include="all"))

# ---------------------------------------------------------
# 2. Clean the Data
# ---------------------------------------------------------

st.header("2. Clean the Data")

missing_values = df.isnull().sum()

st.write("Missing values:")
st.dataframe(
    missing_values.to_frame("Missing Values")
)

if df.isnull().sum().sum() > 0:
    df = df.dropna()
    st.success("Missing rows removed.")
else:
    st.success("No missing values found.")

# ---------------------------------------------------------
# Check Required Columns
# ---------------------------------------------------------

required_columns = [
    "weight",
    "diameter",
    "color",
    "fruit"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required columns: {missing_columns}"
    )
    st.stop()

# ---------------------------------------------------------
# 3. Exploratory Data Analysis
# ---------------------------------------------------------

st.header("3. Exploratory Data Analysis")

st.write("Number of records:", df.shape[0])
st.write("Number of columns:", df.shape[1])

st.subheader("Fruit Distribution")

st.bar_chart(
    df["fruit"].value_counts()
)

# ---------------------------------------------------------
# 4. Feature Engineering / Preparation
# ---------------------------------------------------------

st.header("4. Feature Engineering")

st.write(
    "No new feature is required for this dataset. "
    "The existing features are sufficient for KNN."
)

# ---------------------------------------------------------
# Separate X and Y
# ---------------------------------------------------------

X = df[["weight", "diameter", "color"]]
y = df["fruit"]

st.subheader("X - Input Features")
st.dataframe(X.head())

st.subheader("Y - Target")
st.dataframe(y.head())

# ---------------------------------------------------------
# Encode Color
# ---------------------------------------------------------

color_encoder = LabelEncoder()

X = X.copy()

X["color"] = color_encoder.fit_transform(
    X["color"]
)

st.subheader("Encoded Features")
st.dataframe(X.head())

# ---------------------------------------------------------
# 5. Train Test Split
# ---------------------------------------------------------

st.header("5. Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

st.write(
    "Training records:",
    len(X_train)
)

st.write(
    "Testing records:",
    len(X_test)
)

# ---------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ---------------------------------------------------------
# 6. Train the Model
# ---------------------------------------------------------

st.header("6. Train the KNN Model")

k = st.slider(
    "Select K value",
    min_value=1,
    max_value=10,
    value=3
)

model = KNeighborsClassifier(
    n_neighbors=k
)

model.fit(
    X_train_scaled,
    y_train
)

st.success(
    f"KNN model trained successfully with K = {k}"
)

# ---------------------------------------------------------
# 7. Test the Model
# ---------------------------------------------------------

st.header("7. Test the Model")

predictions = model.predict(
    X_test_scaled
)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

st.dataframe(
    results,
    use_container_width=True
)

# ---------------------------------------------------------
# 8. Results
# ---------------------------------------------------------

st.header("8. Model Results")

# Accuracy

accuracy = accuracy_score(
    y_test,
    predictions
)

st.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

st.dataframe(cm_df)

# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

st.text(report)

# ---------------------------------------------------------
# 9. Predict a New Fruit
# ---------------------------------------------------------

st.header("9. Predict a New Fruit")

col1, col2, col3 = st.columns(3)

with col1:
    weight = st.number_input(
        "Weight",
        min_value=0.0,
        value=150.0
    )

with col2:
    diameter = st.number_input(
        "Diameter",
        min_value=0.0,
        value=7.0
    )

with col3:
    color = st.selectbox(
        "Color",
        color_encoder.classes_
    )

if st.button("Predict Fruit"):

    # Convert color into encoded value

    color_encoded = color_encoder.transform(
        [color]
    )[0]

    # Create new input

    new_fruit = pd.DataFrame(
        [[weight, diameter, color_encoded]],
        columns=[
            "weight",
            "diameter",
            "color"
        ]
    )

    # Scale using the SAME scaler

    new_fruit_scaled = scaler.transform(
        new_fruit
    )

    # Prediction

    prediction = model.predict(
        new_fruit_scaled
    )[0]

    st.success(
        f" Predicted Fruit: {prediction}"
    )

    # Prediction probability

    probabilities = model.predict_proba(
        new_fruit_scaled
    )[0]

    probability_df = pd.DataFrame({
        "Fruit": model.classes_,
        "Probability": probabilities
    })

    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(2)

    st.subheader("Prediction Confidence")

    st.dataframe(
        probability_df,
        use_container_width=True
    )