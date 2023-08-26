import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import shap

# Feature 1: Data Upload and Preview
st.title("ML and Data Visualization App")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if df.isnull().values.any():
        # Remove rows with missing values
        df = df.dropna()
        st.warning("NaN values have been removed:")
        
    
    
    st.write("Data Preview:")
    st.write(df.head())
    
    # Feature 2: Data Visualization
    chart_type = st.selectbox("Select chart type", ['Scatter', 'Box', 'Histogram'])
    x_axis = st.selectbox("Select x-axis:", df.columns)
    y_axis = st.selectbox("Select y-axis:", df.columns)
    
    fig = px.scatter(df, x=x_axis, y=y_axis)  # Default to scatter
    if chart_type == 'Box':
        fig = px.box(df, x=x_axis, y=y_axis)
    elif chart_type == 'Histogram':
        fig = px.histogram(df, x=x_axis)
        
    st.plotly_chart(fig)
    
    # Feature 3: Model Selection and Training
    if len(df) < 2:
        st.warning("Insufficient data for training.")
    else:
        target_var = st.selectbox("Select target variable:", df.columns)
        input_vars = st.multiselect("Select input variables:", df.columns, default=df.columns.drop(target_var).tolist())
        
        if target_var and input_vars:
            X = df[input_vars]
            y = df[target_var]
            
            # One-hot encode categorical variables
            X = pd.get_dummies(X, drop_first=True)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            model_type = st.selectbox("Select model:", ['Linear Regression', 'Random Forest', 'SVM'])
            
            if model_type == 'Linear Regression':
                model = LinearRegression()
            elif model_type == 'Random Forest':
                n_estimators = st.slider("Number of Trees", 10, 100)
                model = RandomForestRegressor(n_estimators=n_estimators)
            elif model_type == 'SVM':
                C = st.slider("C", 0.01, 10.0)
                model = SVR(C=C)
            
            model.fit(X_train, y_train)
            
            # Feature 4: Model Evaluation
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R-squared: {r2}")

# Feature 5: Model Visualization
    # Feature 5: Advanced Model Visualization with SHAP
    st.write("Advanced Model Visualization:")

    explainer = None
    shap_values = None

    if model_type == 'Random Forest':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)[0]  # Taking the first element because it's for the class of interest

    elif model_type == 'Linear Regression':
        explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test)

    if explainer is not None and shap_values is not None:
        # Visualize a single prediction (for the first instance in the test set)
        st.write("SHAP value for a single prediction:")
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True, show=False)
        plt.close(fig)
        st.pyplot(fig)

        # Convert to numpy arrays and ensure they are floats
        shap_values = np.array(shap_values).astype(float)
        X_test_values = X_test.to_numpy().astype(float)

        # Summary plot
        st.write("SHAP Summary Plot:")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test_values, show=False)
        plt.close(fig)
        st.pyplot(fig)
