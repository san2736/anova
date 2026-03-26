import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="ANOVA Analyzer", layout="centered")

st.title("Automatic ANOVA + Tukey Analysis")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()

    dependent = st.selectbox("Select Dependent Variable", columns)

    if st.button("Run Analysis"):

        # check numeric dependent
        if not pd.api.types.is_numeric_dtype(df[dependent]):
            st.error("Dependent variable must be numeric")
        else:
            st.subheader("Results")

            for col in columns:
                if col == dependent:
                    continue

                unique_vals = df[col].nunique()

                # treat as categorical
                if unique_vals <= 10:
                    try:
                        formula = f'{dependent} ~ C({col})'
                        model = ols(formula, data=df).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)

                        p_value = anova_table["PR(>F)"][0]

                        st.write(f"Factor: {col}")
                        st.dataframe(anova_table)

                        # ONLY run Tukey if significant
                        if p_value < 0.05:
                            st.success("Significant → Running Tukey Test")

                            tukey = pairwise_tukeyhsd(
                                endog=df[dependent],
                                groups=df[col],
                                alpha=0.05
                            )

                            tukey_df = pd.DataFrame(
                                tukey._results_table.data[1:],
                                columns=tukey._results_table.data[0]
                            )

                            st.dataframe(tukey_df)

                        else:
                            st.info("Not significant → Tukey skipped")

                    except Exception as e:
                        st.warning(f"Skipped {col} due to error")
