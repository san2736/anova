import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="ANOVA Analysis", layout="centered")

st.title("Two-Way ANOVA Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    st.subheader("Select Variables")

    dependent = st.selectbox("Select Dependent Variable", columns)
    factor1 = st.selectbox("Select First Factor", columns)
    factor2 = st.selectbox("Select Second Factor", columns)

    if st.button("Run Analysis"):

        try:
            # ANOVA
            formula = f'{dependent} ~ {factor1} * {factor2}'
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            st.subheader("ANOVA Table")
            st.dataframe(anova_table)

            # Tukey
            st.subheader("Tukey HSD Test")

            df["group"] = df[factor1].astype(str) + "_" + df[factor2].astype(str)

            tukey = pairwise_tukeyhsd(
                endog=df[dependent],
                groups=df["group"],
                alpha=0.05
            )

            tukey_df = pd.DataFrame(
                tukey._results_table.data[1:],
                columns=tukey._results_table.data[0]
            )

            st.dataframe(tukey_df)

        except Exception as e:
            st.error(f"Error: {e}")
