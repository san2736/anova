import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.title("Advanced ANOVA → Tukey → T-Test Pipeline")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.write("Data Preview:")
    st.dataframe(df.head())

    # Step 1: select continuous variable
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select Continuous Variable", num_cols)

    if target:

        alpha = 0.05

        # Step 2: categorical columns
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        st.write("Categorical columns:", cat_cols)

        # -------- STEP 1: INDIVIDUAL --------
        st.subheader("Step 1: Individual ANOVA")

        significant_vars = []

        for col in cat_cols:
            temp = df[[col, target]].dropna()

            groups = [temp[temp[col] == val][target]
                      for val in temp[col].unique()
                      if len(temp[temp[col] == val]) > 1]

            if len(groups) > 1:
                f, p = stats.f_oneway(*groups)
                st.write(f"{col} → p = {p:.4f}")

                if p < alpha:
                    significant_vars.append(col)

        st.write("Significant after Step 1:", significant_vars)

        # -------- STEP 2: ADDITIVE COMBINATIONS --------
        st.subheader("Step 2: Additive Combinations")

        add_significant = []

        for r in range(2, len(significant_vars) + 1):
            for combo in combinations(significant_vars, r):

                new_col = "_".join(combo)

                temp = df[list(combo) + [target]].dropna()
                temp[new_col] = temp[list(combo)].astype(str).agg("_".join, axis=1)

                groups = [temp[temp[new_col] == val][target]
                          for val in temp[new_col].unique()
                          if len(temp[temp[new_col] == val]) > 1]

                if len(groups) > 1:
                    f, p = stats.f_oneway(*groups)
                    st.write(f"{new_col} → p = {p:.4f}")

                    if p < alpha:
                        add_significant.append(new_col)

        st.write("Significant after Step 2:", add_significant)

        # -------- STEP 3: INTERACTION --------
        st.subheader("Step 3: Interaction (Multiplication-like)")

        final_vars = []

        for col in add_significant:
            temp = df[[target]].copy()
            temp[col] = df[col.split("_")].astype(str).agg("_".join, axis=1)
            temp = temp.dropna()

            groups = [temp[temp[col] == val][target]
                      for val in temp[col].unique()
                      if len(temp[temp[col] == val]) > 1]

            if len(groups) > 1:
                f, p = stats.f_oneway(*groups)
                st.write(f"{col} → p = {p:.4f}")

                if p < alpha:
                    final_vars.append(col)

        st.write("Final selected variables:", final_vars)

        # -------- STEP 4: TUKEY --------
        st.subheader("Step 4: Tukey HSD")

        tukey_results_df = pd.DataFrame()

        for col in final_vars:
            temp = df.copy()
            temp[col] = temp[col.split("_")].astype(str).agg("_".join, axis=1)
            temp = temp[[col, target]].dropna()

            tukey = pairwise_tukeyhsd(
                endog=temp[target],
                groups=temp[col],
                alpha=alpha
            )

            res_df = pd.DataFrame(data=tukey._results_table.data[1:],
                                  columns=tukey._results_table.data[0])

            # Keep only TRUE (reject)
            res_df = res_df[res_df['reject'] == True]

            if not res_df.empty:
                res_df['variable'] = col
                tukey_results_df = pd.concat([tukey_results_df, res_df])

        if tukey_results_df.empty:
            st.warning("No significant Tukey results.")
        else:
            st.dataframe(tukey_results_df)

            # -------- STEP 5: MAX MEAN DIFF --------
            st.subheader("Step 5: Max Mean Difference")

            tukey_results_df['abs_diff'] = tukey_results_df['meandiff'].abs()

            best_row = tukey_results_df.loc[tukey_results_df['abs_diff'].idxmax()]

            st.write("Best Pair:")
            st.write(best_row)

            # -------- STEP 6: T-TEST --------
            st.subheader("Step 6: Final T-Test")

            var = best_row['variable']
            g1 = best_row['group1']
            g2 = best_row['group2']

            temp = df.copy()
            temp[var] = temp[var.split("_")].astype(str).agg("_".join, axis=1)
            temp = temp[[var, target]].dropna()

            group1 = temp[temp[var] == g1][target]
            group2 = temp[temp[var] == g2][target]

            t_stat, p_val = stats.ttest_ind(group1, group2)

            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_val:.4f}")

            if p_val < alpha:
                st.success("Final result is SIGNIFICANT")
            else:
                st.error("Final result is NOT significant")
