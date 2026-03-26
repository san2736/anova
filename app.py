import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.title("Automatic ANOVA + Tukey Analysis")

# Upload file
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.write("Preview of Data:")
    st.dataframe(df.head())

    # Select continuous variable
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    target = st.selectbox("Select Continuous Variable (e.g., balance)", numeric_cols)

    if target:
        st.write(f"Selected variable: {target}")

        # Identify categorical columns
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if len(cat_cols) == 0:
            st.warning("No categorical columns found.")
        else:
            st.write("Categorical columns detected:")
            st.write(cat_cols)

            alpha = 0.05

            results = []

            for col in cat_cols:
                groups = []

                # Drop NA properly
                temp = df[[col, target]].dropna()

                unique_vals = temp[col].unique()

                for val in unique_vals:
                    group = temp[temp[col] == val][target]
                    if len(group) > 1:
                        groups.append(group)

                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)

                    results.append((col, f_stat, p_val))

            # Show ANOVA results
            st.subheader("ANOVA Results")

            for col, f_stat, p_val in results:
                st.write(f"{col} → F = {f_stat:.4f}, p = {p_val:.4f}")

            # Tukey for significant ones
            st.subheader("Tukey Test (Only Significant ANOVA)")

            for col, f_stat, p_val in results:
                if p_val < alpha:
                    st.write(f"Running Tukey for: {col}")

                    temp = df[[col, target]].dropna()

                    tukey = pairwise_tukeyhsd(
                        endog=temp[target],
                        groups=temp[col],
                        alpha=alpha
                    )

                    st.text(tukey.summary())
