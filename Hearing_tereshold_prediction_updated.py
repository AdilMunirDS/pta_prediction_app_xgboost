import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor  # ✅ XGBoost imported

# Load your data
@st.cache_data
def load_data():
    df = pd.read_excel("Research Raw Data.xlsx")
    column_names = ["S#", "Age", "Gender",
                    "PTA_500Hz", "ASSR_500Hz", "PTA_1KHz", "ASSR_1KHz",
                    "PTA_2KHz", "ASSR_2KHz", "PTA_4KHz", "ASSR_4KHz",
                    "PTA_500Hz", "ASSR_500Hz", "PTA_1KHz", "ASSR_1KHz",
                    "PTA_2KHz", "ASSR_2KHz", "PTA_4KHz", "ASSR_4KHz"]
    df.columns = column_names
    df.drop("S#", axis=1, inplace=True)
    df1 = df.iloc[:, :10]
    df2 = df.iloc[:, [0, 1, 10, 11, 12, 13, 14, 15, 16, 17]]
    df3 = pd.concat([df1, df2], ignore_index=True)

    df3.replace('-', np.nan)
    df3 = df3[["Age", "Gender", "PTA_500Hz", "PTA_1KHz", "PTA_2KHz", "PTA_4KHz",
               "ASSR_500Hz", "ASSR_1KHz", "ASSR_2KHz", "ASSR_4KHz"]]

    exclude_cols = ['Gender']
    cols_to_convert = df.columns.difference(exclude_cols)
    df3[cols_to_convert] = df3[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    df_cleaned = df3.dropna(thresh=df3.shape[1] - 2, ignore_index=True)
    return df_cleaned

df = load_data()

def handle_missing_values(row):
    for i in range(len(row)):
        if i < 6:
            row[i] = row[i] if not pd.isna(row[i]) else 110
        else:
            row[i] = row[i] if not pd.isna(row[i]) else 100
    return row

df.iloc[:, :] = df.apply(handle_missing_values, axis=1)

# Normalize the data
columns_to_scale = ["Age", "ASSR_500Hz", "ASSR_1KHz", "ASSR_2KHz", "ASSR_4KHz"]
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Prepare features and target
X = df[["Age", "Gender", "ASSR_500Hz", "ASSR_1KHz", "ASSR_2KHz", "ASSR_4KHz"]]
y = df[["PTA_500Hz", "PTA_1KHz", "PTA_2KHz", "PTA_4KHz"]]
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

st.title("PTA Threshold Prediction App")

# Input
age = st.number_input("Age", min_value=0, max_value=120, value=10)
gender = st.selectbox("Gender", ["Male", "Female"], index=0)
ASSR_500Hz = st.number_input("ASSR_500Hz", value=50)
ASSR_1KHz = st.number_input("ASSR_1KHz", value=60)
ASSR_2KHz = st.number_input("ASSR_2KHz", value=60)
ASSR_4KHz = st.number_input("ASSR_4KHz", value=70)

# Model selection
model_option = st.selectbox("Select a model", [
    "LinearRegression", "SVM", "RandomForest", "DecisionTree", "KNN", "XGBoost"])

# Hyperparameter tuning
params = {}
if model_option == "SVM":
    params = {
        'estimator__C': [st.slider("C (Regularization)", 0.01, 10.0, 1.0)],
        'estimator__kernel': [st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])]
    }
elif model_option == "RandomForest":
    params = {
        'n_estimators': [st.slider("Number of Trees", 10, 200, 100)],
        'max_depth': [st.slider("Max Depth", 1, 20, 10)]
    }
elif model_option == "DecisionTree":
    params = {
        'max_depth': [st.slider("Max Depth", 1, 20, 10)]
    }
elif model_option == "KNN":
    params = {
        'estimator__n_neighbors': [st.slider("Number of Neighbors", 1, 30, 5)]
    }
elif model_option == "XGBoost":
    params = {
        'estimator__n_estimators': [st.slider("Number of Estimators", 50, 300, 100)],
        'estimator__max_depth': [st.slider("Max Depth", 1, 10, 3)],
        'estimator__learning_rate': [st.slider("Learning Rate", 0.01, 0.5, 0.1)]
    }

# Prepare input
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "ASSR_500Hz": [ASSR_500Hz],
    "ASSR_1KHz": [ASSR_1KHz],
    "ASSR_2KHz": [ASSR_2KHz],
    "ASSR_4KHz": [ASSR_4KHz]
})

input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])
input_data = pd.get_dummies(input_data, columns=['Gender'], drop_first=True).reindex(columns=X.columns, fill_value=0)

# Predict
if st.button("Predict"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_option == "LinearRegression":
        model = LinearRegression()
    elif model_option == "SVM":
        model = MultiOutputRegressor(SVR())
    elif model_option == "RandomForest":
        model = RandomForestRegressor()
    elif model_option == "DecisionTree":
        model = DecisionTreeRegressor()
    elif model_option == "KNN":
        model = MultiOutputRegressor(KNeighborsRegressor())
    elif model_option == "XGBoost":
        model = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', random_state=42))

    # Grid search if applicable
    if params:
        grid_search = GridSearchCV(model, param_grid=params, cv=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(input_data)
    st.markdown(f"""
    **Predicted PTA Thresholds:**
    - PTA_500Hz: {y_pred[0][0]:.2f}
    - PTA_1KHz: {y_pred[0][1]:.2f}
    - PTA_2KHz: {y_pred[0][2]:.2f}
    - PTA_4KHz: {y_pred[0][3]:.2f}
    """)

    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    st.write(f"Mean Absolute Error: {mae:.2f}")

    error_threshold = 10
    dif = np.abs(y_test - y_test_pred)
    accuracy = (dif < error_threshold).mean(axis=0)
    overall_accuracy = (accuracy.mean()) * 100
    st.write(f"Accuracy (within {error_threshold} dB): {overall_accuracy:.2f}")
