import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

st.set_page_config(page_title="ANOVA Analyzer", layout="centered")

st.title("Automatic ANOVA Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    # Select dependent variable
    dependent = st.selectbox("Select Dependent Variable", columns)

    if st.button("Run Analysis"):

        # SMART CHECK 1: dependent must be numeric
        if not pd.api.types.is_numeric_dtype(df[dependent]):
            st.error("Dependent variable must be numeric")
        else:
            st.subheader("ANOVA Results")

            results_found = False

            for col in columns:
                if col == dependent:
                    continue

                unique_vals = df[col].nunique()

                # SMART CHECK 2: treat as categorical only if limited unique values
                if unique_vals <= 10:
                    try:
                        formula = f'{dependent} ~ C({col})'
                        model = ols(formula, data=df).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)

                        st.write(f"Factor: {col}")
                        st.dataframe(anova_table)

                        results_found = True

                    except:
                        continue

            if not results_found:
                st.warning("No suitable categorical variables found (need <=10 unique values)")
