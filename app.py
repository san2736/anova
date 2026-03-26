import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.title("Statistical Analysis App (ANOVA + Tukey)")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select continuous variable
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select Continuous Variable", numeric_cols)

    if target:
        st.write(f"Selected variable: {target}")

        # Identify categorical columns
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if len(cat_cols) == 0:
            st.warning("No categorical variables found.")
        else:
            alpha = 0.05

            st.header("ANOVA Analysis")

            for col in cat_cols:

                st.subheader(f"Testing: {target} vs {col}")

                temp = df[[col, target]].dropna()

                groups = []
                labels = []

                for val in temp[col].unique():
                    group = temp[temp[col] == val][target]
                    if len(group) > 1:
                        groups.append(group)
                        labels.append(val)

                if len(groups) <= 1:
                    st.write("Not enough groups to perform ANOVA.")
                    continue

                # Hypothesis
                st.write("H0: All group means are equal")
                st.write("H1: At least one group mean is different")

                # ANOVA
                f_stat, p_val = stats.f_oneway(*groups)

                st.write(f"F-statistic: {f_stat:.4f}")
                st.write(f"p-value: {p_val:.4f}")

                # Decision
                if p_val < alpha:
                    st.success("Reject H0 → Significant difference exists")
                    
                    # Tukey Test
                    st.write("Performing Tukey Test...")

                    tukey = pairwise_tukeyhsd(
                        endog=temp[target],
                        groups=temp[col],
                        alpha=alpha
                    )

                    result_df = pd.DataFrame(
                        data=tukey.summary().data[1:], 
                        columns=tukey.summary().data[0]
                    )

                    st.dataframe(result_df)

                    # Interpretation
                    st.write("Interpretation:")

                    significant_pairs = result_df[result_df["reject"] == True]

                    if len(significant_pairs) == 0:
                        st.write("No significant pairwise differences found.")
                    else:
                        for _, row in significant_pairs.iterrows():
                            st.write(
                                f"{row['group1']} vs {row['group2']} → significantly different"
                            )

                else:
                    st.error("Fail to Reject H0 → No significant difference")
