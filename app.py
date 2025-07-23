import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(layout="wide", page_title="Porosity-Permeability Correlation (Semi-Log)")

    st.title("🧮 Porosity-Permeability Relationship Analysis")
    st.markdown("""
    Upload your CSV file to explore the empirical correlation between porosity and permeability.
    Given the typical exponential relationship in petrophysics, a **log-linear regression**
    (linear regression of Porosity vs. Log(Permeability)) is performed to derive the correlation.
    """)

    st.sidebar.header("⚙️ Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Uploaded Data Preview")
            st.write(df.head())

            all_columns = df.columns.tolist()
            
            # Smart default column selection
            porosity_options = [col for col in all_columns if 'porosity' in col.lower()]
            permeability_options = [col for col in all_columns if 'permeability' in col.lower()]

            porosity_col_default = 'Porosity' if 'Porosity' in all_columns else (porosity_options[0] if porosity_options else all_columns[0])
            permeability_col_default = 'Permeability' if 'Permeability' in all_columns else (permeability_options[0] if permeability_options else (all_columns[1] if len(all_columns) > 1 else all_columns[0]))

            # Adjust default index for selectbox
            porosity_default_idx = all_columns.index(porosity_col_default) if porosity_col_default in all_columns else 0
            permeability_default_idx = all_columns.index(permeability_col_default) if permeability_col_default in all_columns else (1 if len(all_columns)>1 else 0)

            porosity_col = st.sidebar.selectbox("Select Porosity Column (X-axis)", all_columns, index=porosity_default_idx)
            permeability_col = st.sidebar.selectbox("Select Permeability Column (Y-axis)", all_columns, index=permeability_default_idx)

            if porosity_col and permeability_col:
                df = df[[porosity_col, permeability_col]].dropna()
                df.columns = ['Porosity', 'Permeability'] # Standardize column names
                
                # Assume Porosity might be in fraction, convert to percentage if max is <= 1.0
                if df['Porosity'].max() <= 1.0 and df['Porosity'].min() >= 0.0:
                    st.sidebar.warning("Porosity values appear to be in fraction (0-1). Multiplying by 100 for percentage.")
                    df['Porosity'] = df['Porosity'] * 100

                if df.empty:
                    st.error("Selected columns resulted in an empty dataset after dropping missing values. Please check your data or column selection.")
                    return
                
                st.success(f"Using '{porosity_col}' as Porosity (%) and '{permeability_col}' as Permeability (mD).")
            else:
                st.warning("Please select both Porosity and Permeability columns.")
                return
        except Exception as e:
            st.error(f"Error reading CSV file or processing columns: {e}. Please ensure it's a valid CSV and columns contain numeric data.")
            return
    else:
        st.info("⬆️ Please upload a CSV file in the sidebar to begin.")
        # Optionally, load synthetic data as a default if no file is uploaded.
        # This can be useful for initial demo, but the user requested removal of choice.
        # df = generate_synthetic_data_for_log_plot(num_samples=1000, noise_std_dev_ln_perm=0.7)
        # st.info("No CSV uploaded, loading a default synthetic dataset for demonstration.")


    if df is not None and not df.empty:
        # Handle zero or negative permeability values by replacing with a small positive number
        if (df['Permeability'] <= 0).any():
            st.warning("Permeability values less than or equal to zero found. Replacing with a small positive value (0.0001) for logarithmic transformation.")
            df['Permeability'] = df['Permeability'].apply(lambda x: max(x, 0.0001))

        df['Log_Permeability'] = np.log(df['Permeability'])

        st.subheader("📈 Data Statistics")
        st.write(df[['Porosity', 'Permeability']].describe())

        # Log-Linear Regression
        st.subheader("✨ Log-Linear Regression Analysis")
        
        X = df[['Porosity']]
        y = df['Log_Permeability']

        if len(X) < 2:
            st.warning("Not enough data points to perform linear regression. Please ensure at least 2 data points.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_log = model.predict(X_test)
        r2_log_transformed = r2_score(y_test, y_pred_log)

        # Coefficients for the linearized model: ln(Permeability) = slope * Porosity + intercept
        slope = model.coef_[0]
        intercept = model.intercept_
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Coefficient (Slope for ln(Permeability) vs. Porosity)", value=f"{slope:.4f}")
        with col2:
            st.metric(label="Intercept (for ln(Permeability))", value=f"{intercept:.4f}")
            
        st.metric(label="R-squared (R² on Log-Transformed Permeability)", value=f"{r2_log_transformed:.4f}", help="R-squared for the linear fit of Porosity vs. ln(Permeability).")

        # Display the empirical correlation equation more clearly
        st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Derived Empirical Correlation Equation</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        The linear regression model in the log-transformed space is:
        
        $$\\ln(\\text{{Permeability}}) = ({slope:.4f} \\times \\text{{Porosity (\\%)}}) + {intercept:.4f}$$
        
        Transforming this back to an exponential form for direct use:
        
        $$\\text{{Permeability}} = e^{{({slope:.4f} \\times \\text{{Porosity (\\%)}}) + {intercept:.4f}}}$$
        
        Which can also be written as:
        
        $$\\text{{Permeability}} = {np.exp(intercept):.4f} \\times e^{{({slope:.4f} \\times \\text{{Porosity (\\%)}})}}$$
        
        Where:
        - $\\text{{Permeability}}$ is in mD
        - $\\text{{Porosity}}$ is in %
        - $e$ is Euler's number (approximately 2.71828)
        """)

        # Plotting - Semi-Log Scale
        st.subheader("📉 Regression Plot: Porosity vs. Permeability (Semi-Log Scale)")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.scatterplot(x='Porosity', y='Permeability', data=df, ax=ax, label='Actual Data Points', alpha=0.7)
        
        # Plotting the regression line (straight on log-y scale)
        porosity_min = df['Porosity'].min()
        porosity_max = df['Porosity'].max()
        x_line = np.linspace(porosity_min, porosity_max, 100).reshape(-1, 1) # Create smooth line for plotting
        y_line_log_predicted = model.predict(x_line)
        y_line_predicted_exp = np.exp(y_line_log_predicted) # Transform back to linear Permeability values
        
        ax.plot(x_line, y_line_predicted_exp, color='red', linestyle='-', label='Log-Linear Regression Line', linewidth=2)
        
        ax.set_title("Porosity vs. Permeability with Log-Linear Regression Fit")
        ax.set_xlabel("Core Porosity (%)")
        ax.set_ylabel("Permeability (mD)")
        ax.legend()
        ax.grid(True, which="both", ls="--", c='0.7', alpha=0.6) 
        ax.set_yscale('log') # Set y-axis to logarithmic scale
        
        
        st.pyplot(fig)

        # Prediction section
        st.subheader("🔮 Make a Prediction")
        
        min_porosity_val = float(df['Porosity'].min())
        max_porosity_val = float(df['Porosity'].max())
        
        input_porosity = st.number_input(
            "Enter Porosity value (%):", 
            min_value=min_porosity_val, 
            max_value=max_porosity_val, 
            value=min_porosity_val + (max_porosity_val - min_porosity_val) / 2,
            step=(max_porosity_val - min_porosity_val) / 100,
            format="%.4f"
        )
        
        predicted_log_permeability = model.predict(np.array([[input_porosity]]))[0]
        predicted_permeability = np.exp(predicted_log_permeability)
        
        st.markdown(f"**Predicted Permeability (mD) for Porosity `{input_porosity:.4f}%`:** `{predicted_permeability:.4f}`")

    st.sidebar.markdown("---")
    st.sidebar.info("Developed with ❤️ using Streamlit")


if __name__ == "__main__":
    main()