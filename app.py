import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.title("Smart ANOVA → Tukey → T-Test")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.dataframe(df.head())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select Continuous Variable", num_cols)

    if target:

        alpha = 0.05

        # Identify categorical only
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        st.write("Categorical Columns:", cat_cols)

        # -------- STEP 1 --------
        st.subheader("Step 1: Individual ANOVA")

        significant_vars = []

        for col in cat_cols:
            temp = df[[col, target]].dropna()

            groups = [temp[temp[col] == v][target]
                      for v in temp[col].unique()
                      if len(temp[temp[col] == v]) > 1]

            if len(groups) > 1:
                f, p = stats.f_oneway(*groups)
                st.write(f"{col} → p = {p:.4f}")

                if p < alpha:
                    significant_vars.append(col)

        st.write("After Step 1:", significant_vars)

        # -------- STEP 2 (ADDITIVE WITH PRUNING) --------
        st.subheader("Step 2: Additive Pruning")

        var_score = {v: 0 for v in significant_vars}

        for combo in combinations(significant_vars, 2):

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
                    for v in combo:
                        var_score[v] += 1
                else:
                    for v in combo:
                        var_score[v] -= 1

        # Keep only strong variables
        pruned_vars = [v for v in var_score if var_score[v] > 0]

        st.write("After Additive Pruning:", pruned_vars)

        # -------- STEP 3 (INTERACTION WITH PRUNING) --------
        st.subheader("Step 3: Interaction Pruning")

        var_score2 = {v: 0 for v in pruned_vars}

        for combo in combinations(pruned_vars, 2):

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
                    for v in combo:
                        var_score2[v] += 1
                else:
                    for v in combo:
                        var_score2[v] -= 1

        final_vars = [v for v in var_score2 if var_score2[v] > 0]

        st.write("Final Variables:", final_vars)

        # -------- STEP 4: TUKEY --------
        st.subheader("Step 4: Tukey")

        tukey_df = pd.DataFrame()

        for combo in combinations(final_vars, 2):

            col = "_".join(combo)

            temp = df.copy()
            temp[col] = temp[list(combo)].astype(str).agg("_".join, axis=1)
            temp = temp[[col, target]].dropna()

            tukey = pairwise_tukeyhsd(temp[target], temp[col])

            res = pd.DataFrame(tukey._results_table.data[1:],
                               columns=tukey._results_table.data[0])

            res = res[res['reject'] == True]

            if not res.empty:
                res['variable'] = col
                tukey_df = pd.concat([tukey_df, res])

        if tukey_df.empty:
            st.warning("No Tukey results")
        else:
            st.dataframe(tukey_df)

            # -------- STEP 5 --------
            st.subheader("Step 5: Max Mean Difference")

            tukey_df['abs_diff'] = tukey_df['meandiff'].abs()
            best = tukey_df.loc[tukey_df['abs_diff'].idxmax()]

            st.write(best)

            # -------- STEP 6 --------
            st.subheader("Step 6: Final T-Test")

            var = best['variable']
            g1 = best['group1']
            g2 = best['group2']

            temp = df.copy()
            cols = var.split("_")

            temp[var] = temp[cols].astype(str).agg("_".join, axis=1)
            temp = temp[[var, target]].dropna()

            grp1 = temp[temp[var] == g1][target]
            grp2 = temp[temp[var] == g2][target]

            t, p = stats.ttest_ind(grp1, grp2)

            st.write(f"T = {t:.4f}, p = {p:.4f}")

            if p < alpha:
                st.success("SIGNIFICANT")
            else:
                st.error("NOT SIGNIFICANT")
