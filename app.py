"""
app.py - Streamlit dashboard
Regression Diagnostics and Influential Observation Analysis (California Housing)

Run locally:   streamlit run app.py
All statistics come from sml_core.py, the same module the notebook uses.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import sml_core as core

st.set_page_config(page_title="Regression Diagnostics", page_icon="📊", layout="wide")

# ---------- light custom styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
div[data-testid="stMetric"] {background: rgba(100,120,200,0.08); border: 1px solid rgba(100,120,200,0.25);
    padding: 12px 16px; border-radius: 12px;}
.hero {background: linear-gradient(120deg,#1e3a8a,#6d28d9); color: white; padding: 22px 28px;
    border-radius: 14px; margin-bottom: 18px;}
.hero h1 {color: white; margin: 0; font-size: 1.9rem;}
.hero p {margin: 6px 0 0 0; opacity: .92;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading data and fitting models ...")
def get_results(rule: str):
    """Runs the full methodology once per flagging rule and caches it."""
    return core.run_analysis(rule=rule)


def csv_bytes(df: pd.DataFrame, index=False) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


def explain(text: str, title: str = "What does this mean?"):
    with st.expander(title, expanded=False):
        st.markdown(text)


# ---------- sidebar ----------
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Go to", [
    "1. Project Overview", "2. Dataset Exploration", "3. Regression Model Performance",
    "4. Regression Diagnostics", "5. Influential Observation Analysis",
    "6. Original vs Sensitivity Model", "7. New Prediction"])
st.sidebar.markdown("---")
rule = st.sidebar.selectbox("Flagging rule for the sensitivity analysis",
                            list(core.FLAG_RULES), format_func=lambda k: core.FLAG_RULES[k], index=0,
                            help="'Cook' is the default used in the notebook and the report.")
st.sidebar.caption(f"Random seed = {core.RANDOM_STATE} · test size = {core.TEST_SIZE:.0%}")

try:
    res = get_results(rule)
except Exception as e:
    st.error(f"Could not load the data: {e}")
    st.stop()

df, orig, sens = res["df"], res["orig"], res["sens"]
infl, th, comp, diag = res["infl"], res["thresholds"], res["comp"], res["diag"]
flagged = res["flagged"]

# =============================================================================
if section.startswith("1"):
    st.markdown("""<div class="hero"><h1>Regression Diagnostics & Influential Observation Analysis</h1>
    <p>Real-world prediction of California house values with a statistically audited linear model</p></div>""",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric("Features", len(core.FEATURES))
    c3.metric("Test R²", f"{res['metrics']['test']['R2']:.4f}")
    c4.metric("Flagged (potentially influential)", f"{int(flagged.sum()):,}")
    st.subheader("Problem statement")
    st.write("A regression model can look good on average and still be shaped by a handful of unusual observations. "
             "This project fits a multiple linear regression, checks its assumptions, finds outliers, high-leverage "
             "points and influential observations, and measures how much they change coefficients and predictions.")
    st.subheader("Methodology")
    st.markdown("""
1. Load California Housing (scikit-learn) → reproducible 80/20 split.
2. Fit OLS (statsmodels) on the **training set only**.
3. Check assumptions: residual plots, Q-Q plot, Breusch-Pagan, Jarque-Bera, VIF.
4. Influence analysis on the training set: Cook's D, leverage, studentized residuals, DFFITS.
5. Investigate flagged points, refit **without** them (sensitivity model).
6. Compare both models on the **same untouched test set**.
""")
    st.info(f"**Target:** `{core.TARGET}` = median house value in units of **$100,000** (e.g. 2.5 = $250,000). "
            f"The target is **capped at {core.TARGET_CAP}** ($500,000): {res['overview']['pct_at_cap']:.2f}% of "
            "rows sit exactly at the cap.")

# =============================================================================
elif section.startswith("2"):
    st.header("Dataset Exploration")
    ov = res["overview"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{ov['shape'][0]:,}")
    c2.metric("Columns", ov["shape"][1])
    c3.metric("Missing values", int(ov["missing"].sum().iloc[0]))
    c4.metric("Duplicate rows", ov["n_duplicates"])
    tab1, tab2, tab3, tab4 = st.tabs(["Preview & statistics", "Distributions", "Correlation", "Feature vs target"])
    with tab1:
        n = st.slider("Rows to preview", 5, 100, 10)
        st.dataframe(df.head(n))
        st.write("**Descriptive statistics**")
        st.dataframe(ov["describe"].round(3))
        st.write("**Data types**")
        st.dataframe(ov["dtypes"].T)
        st.download_button("⬇ Download dataset (CSV)", csv_bytes(df), "california_housing.csv", "text/csv")
    with tab2:
        col = st.selectbox("Variable", [core.TARGET] + core.FEATURES)
        fig = px.histogram(df, x=col, nbins=60, marginal="box", title=f"Distribution of {col}")
        if col == core.TARGET:
            fig.add_vline(x=core.TARGET_CAP, line_dash="dash", line_color="red", annotation_text="cap = 5.0")
        st.plotly_chart(fig)
    with tab3:
        fig = px.imshow(df.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation heatmap")
        fig.update_layout(height=600)
        st.plotly_chart(fig)
    with tab4:
        col = st.selectbox("Feature", core.FEATURES, key="fvt")
        s = df.sample(min(4000, len(df)), random_state=core.RANDOM_STATE)
        fig = px.scatter(s, x=col, y=core.TARGET, opacity=0.35, trendline="ols",
                         title=f"{col} vs {core.TARGET} (sample of 4,000 rows)")
        st.plotly_chart(fig)
    st.subheader("Statistical observations (computed from the data)")
    for o in res["eda_notes"]:
        st.markdown(f"- {o}")

# =============================================================================
elif section.startswith("3"):
    st.header("Regression Model Performance")
    m = res["metrics"]
    st.caption("Model: OLS fitted on the training set. Test metrics use data the model has never seen.")
    c = st.columns(5)
    c[0].metric("Test R²", f"{m['test']['R2']:.4f}")
    c[1].metric("Adj. R² (train)", f"{orig.rsquared_adj:.4f}")
    c[2].metric("Test MAE", f"{m['test']['MAE']:.4f}")
    c[3].metric("Test MSE", f"{m['test']['MSE']:.4f}")
    c[4].metric("Test RMSE", f"{m['test']['RMSE']:.4f}")
    st.write(pd.DataFrame(m).T.round(4).rename_axis("Split"))
    st.success(res["text"]["model"])
    st.subheader("Coefficients")
    ct = core.coefficient_table(orig)
    st.dataframe(ct.round(5))
    st.download_button("⬇ Coefficients (CSV)", csv_bytes(ct, index=True), "ols_coefficients.csv", "text/csv")
    explain("""
- **Coefficient**: expected change in house value (×$100,000) for a 1-unit increase in that feature, other features fixed.
- **p-value**: probability of seeing such a coefficient if the true effect were 0. With ~16,000 rows even tiny effects are 'significant'.
- **R²**: share of variance explained. **Adjusted R²** penalises useless features.
- **RMSE** is in the same units as the target; MAE is the average absolute error.""")
    which = st.radio("Show", ["Test set", "Training set"], horizontal=True)
    if which == "Test set":
        yt, yp = np.asarray(res["y_test"]), res["test_pred"]
    else:
        yt, yp = np.asarray(res["y_train"]), np.asarray(orig.fittedvalues)
    idx = np.random.RandomState(core.RANDOM_STATE).choice(len(yt), min(4000, len(yt)), replace=False)
    fig = px.scatter(x=yt[idx], y=yp[idx], opacity=0.35, labels={"x": "Actual", "y": "Predicted"},
                     title=f"Actual vs predicted ({which.lower()}, sample of {len(idx):,})")
    fig.add_trace(go.Scatter(x=[yt.min(), yt.max()], y=[yt.min(), yt.max()], mode="lines",
                             name="Perfect fit", line=dict(color="red", dash="dash")))
    st.plotly_chart(fig)
    with st.expander("Full statsmodels OLS summary"):
        st.code(orig.summary().as_text())

# =============================================================================
elif section.startswith("4"):
    st.header("Regression Diagnostics")
    resid, fitted = np.asarray(orig.resid), np.asarray(orig.fittedvalues)
    idx = np.random.RandomState(core.RANDOM_STATE).choice(len(resid), min(5000, len(resid)), replace=False)
    t1, t2, t3, t4, t5, t6 = st.tabs(["Residuals vs fitted", "Histogram", "Q-Q plot", "Breusch-Pagan",
                                       "Jarque-Bera", "VIF"])
    with t1:
        fig = px.scatter(x=fitted[idx], y=resid[idx], opacity=0.35,
                         labels={"x": "Fitted", "y": "Residual"}, title="Residuals vs fitted (sample of 5,000)")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig)
        explain("**Checks:** linearity and constant variance. **Good:** a flat, even band around 0. "
                "**Bad:** curves (non-linearity) or a funnel (heteroscedasticity). A diagonal band at the "
                "top-right is caused by the capped target.")
    with t2:
        fig = px.histogram(x=resid, nbins=80, histnorm="probability density", title="Residual distribution")
        xs = np.linspace(resid.min(), resid.max(), 300)
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, resid.mean(), resid.std(ddof=1)),
                                 name="Normal curve", line=dict(color="red")))
        st.plotly_chart(fig)
        explain("**Checks:** whether errors look bell-shaped. Heavy tails or skew mean some predictions are far off.")
    with t3:
        (osm, osr), (slope, icpt, _) = stats.probplot(resid, dist="norm")
        j = np.random.RandomState(1).choice(len(osm), min(5000, len(osm)), replace=False)
        fig = px.scatter(x=osm[j], y=osr[j], opacity=0.4, labels={"x": "Theoretical quantiles", "y": "Sample quantiles"},
                         title="Normal Q-Q plot")
        fig.add_trace(go.Scatter(x=osm, y=slope * osm + icpt, name="Reference", line=dict(color="red")))
        st.plotly_chart(fig)
        explain("**Checks:** normality of residuals. Points on the line = normal. Ends curving away = heavy tails.")
    with t4:
        c1, c2 = st.columns(2)
        c1.metric("LM statistic", f"{diag['bp']['LM']:.2f}")
        c2.metric("p-value", f"{diag['bp']['LM_p']:.3g}")
        (st.error if diag["bp"]["LM_p"] < core.ALPHA else st.success)(res["text"]["bp"])
        explain("H0: constant variance (homoscedasticity). A small p-value (<0.05) means we reject H0. "
                "Why it matters: standard errors and p-values assume constant variance.")
    with t5:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("JB statistic", f"{diag['jb']['JB']:.1f}")
        c2.metric("p-value", f"{diag['jb']['p']:.3g}")
        c3.metric("Skewness", f"{diag['jb']['skew']:.2f}")
        c4.metric("Kurtosis", f"{diag['jb']['kurtosis']:.2f}")
        (st.error if diag["jb"]["p"] < core.ALPHA else st.success)(res["text"]["jb"])
        explain("H0: residuals are normal (skewness 0, kurtosis 3). JB = n/6·(S² + (K−3)²/4). "
                "Very large samples make the test hypersensitive, so combine it with the Q-Q plot.")
    with t6:
        vif = diag["vif"]
        fig = px.bar(vif, x="VIF", y="Feature", orientation="h", title="Variance Inflation Factor")
        fig.add_vline(x=5, line_dash="dash", line_color="orange")
        fig.add_vline(x=10, line_dash="dash", line_color="red")
        st.plotly_chart(fig)
        st.dataframe(vif.round(3))
        st.info(res["text"]["vif"])
        explain("VIF_j = 1/(1 − R²_j), where R²_j comes from regressing feature j on all other features. "
                "Rules of thumb: >5 moderate, >10 serious multicollinearity.")

# =============================================================================
elif section.startswith("5"):
    st.header("Influential Observation Analysis")
    explain("""
- **Outlier**: unusual **y** given its x-values → large residual (|studentized residual| > 3).
- **High-leverage point**: unusual **x**-values (far from the centre of the predictors) → large hat value h_ii.
- **Influential observation**: removing it noticeably changes the fitted model. Usually it is an outlier **and/or** high-leverage.
  A point can be high-leverage but NOT influential if it lies on the trend.
- **Cook's D** = (e_i² / (p·s²)) · h_ii / (1−h_ii)². **DFFITS** = t_i·√(h_ii/(1−h_ii)).""", "Outlier vs leverage vs influential (read me)")
    c = st.columns(4)
    c[0].metric("Cook's D > 4/n", f"{int(infl.Flag_Cook.sum()):,}")
    c[1].metric("Leverage > 2p/n", f"{int(infl.Flag_Leverage.sum()):,}")
    c[2].metric("|Stud. resid| > 3", f"{int(infl.Flag_Outlier.sum()):,}")
    c[3].metric("|DFFITS| > 2√(p/n)", f"{int(infl.Flag_DFFITS.sum()):,}")
    st.caption(f"Thresholds → n = {th['n']:,}, p = {th['p']}: Cook {th['cook']:.6f} · leverage {th['leverage']:.5f} · "
               f"studentized ±{th['studentized']:.0f} · DFFITS ±{th['dffits']:.4f}")
    st.warning(res["text"]["influence"])
    d = infl.sort_values("Obs_ID").reset_index(drop=True)
    d["Position"] = d.index
    d["Flag"] = np.where(d["Flag_Cook"], "Cook's D > 4/n", "Below threshold")
    fig = px.scatter(d, x="Position", y="Cook_D", color="Flag", opacity=0.7, hover_data=["Obs_ID", "Leverage", "Studentized_Resid"],
                     color_discrete_map={"Cook's D > 4/n": "crimson", "Below threshold": "steelblue"}, title="Cook's distance")
    fig.add_hline(y=th["cook"], line_dash="dash", line_color="orange", annotation_text="4/n")
    st.plotly_chart(fig)
    fig = px.scatter(infl, x="Leverage", y="Studentized_Resid", color="Type", size=infl["Cook_D"].clip(upper=0.05) + 0.002,
                     opacity=0.6, hover_data=["Obs_ID", "Cook_D", "Actual", "Fitted"],
                     title="Leverage vs studentized residual (bubble size ~ Cook's D)")
    fig.add_vline(x=th["leverage"], line_dash="dash", line_color="green")
    fig.add_hline(y=th["studentized"], line_dash="dash", line_color="purple")
    fig.add_hline(y=-th["studentized"], line_dash="dash", line_color="purple")
    st.plotly_chart(fig)
    st.subheader("Ranking of the most influential observations")
    topn = st.slider("Show top N", 10, 200, 25, step=5)
    show_cols = ["Cook_Rank", "Obs_ID", "Cook_D", "Leverage", "Studentized_Resid", "DFFITS", "Actual", "Fitted", "Type"] + core.FEATURES
    st.dataframe(infl[show_cols].head(topn).round(4))
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Full influence table (CSV)", csv_bytes(infl), "influence_table.csv", "text/csv")
    c2.download_button("⬇ Flagged observations (CSV)", csv_bytes(infl[flagged]), "flagged_observations.csv", "text/csv")
    st.subheader("Investigating the flagged observations")
    cc1, cc2 = st.columns([1, 2])
    cc1.dataframe(res["type_counts"].round(2))
    cc2.dataframe(res["profile"].round(3))
    st.caption("Flagged ≠ wrong. Compare the profiles: flagged points are typically unusual blocks (e.g. extreme occupancy "
               "or room counts, or the capped price), which are real data and may need a better model rather than deletion.")

# =============================================================================
elif section.startswith("6"):
    st.header("Original vs Sensitivity Model")
    st.caption(f"Sensitivity model = OLS refit without the {int(flagged.sum()):,} flagged training observations "
               f"({core.FLAG_RULES[rule]}). Both models are scored on the SAME untouched test set.")
    tm = comp["test_metrics"]
    c = st.columns(4)
    for col, m in zip(c, ["R2", "MAE", "MSE", "RMSE"]):
        col.metric(f"Test {m}", f"{tm.loc[m, 'Sensitivity']:.4f}", f"{tm.loc[m, 'Change']:+.4f} vs original",
                   delta_color="off")
    st.dataframe(tm.round(5))
    st.success(res["text"]["comparison"])
    mm = tm.drop(index="R2").reset_index().melt(id_vars="index", value_vars=["Original", "Sensitivity"])
    fig = px.bar(mm, x="index", y="value", color="variable", barmode="group", title="Test error metrics (lower is better)",
                 labels={"index": "Metric", "value": "Value", "variable": "Model"})
    st.plotly_chart(fig)
    st.subheader("Coefficient comparison")
    cf = comp["coef"]
    st.dataframe(cf.round(5))
    cc = cf.drop(index="const")
    fig = px.bar(cc.reset_index(), x="index", y="Shift_in_SE", color="Sens_Inside_Orig_CI",
                 color_discrete_map={True: "seagreen", False: "crimson"},
                 labels={"index": "Feature", "Shift_in_SE": "Shift (original std. errors)", "Sens_Inside_Orig_CI": "Inside original 95% CI"},
                 title="Coefficient shift measured in original standard errors (|shift| > 1.96 = outside 95% CI)")
    fig.add_hline(y=1.96, line_dash="dash", line_color="gray")
    fig.add_hline(y=-1.96, line_dash="dash", line_color="gray")
    st.plotly_chart(fig)
    st.subheader("Is the conclusion sensitive to the flagging rule?")
    st.dataframe(res["sweep"].round(4))
    st.caption("Test metrics here are only reported for transparency; they are never used to choose a rule (that would leak the test set).")
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Metrics (CSV)", csv_bytes(tm, index=True), "test_metrics_comparison.csv", "text/csv")
    c2.download_button("⬇ Coefficients (CSV)", csv_bytes(cf, index=True), "coefficient_comparison.csv", "text/csv")
    c3.download_button("⬇ Rule sweep (CSV)", csv_bytes(res["sweep"]), "threshold_sensitivity.csv", "text/csv")
    st.download_button("⬇ Full results report (Markdown)", core.build_results_markdown(res).encode("utf-8"),
                       "project_results.md", "text/markdown")

# =============================================================================
else:
    st.header("New Prediction")
    st.caption("Enter the characteristics of a census block group. Prediction = OLS fitted on the training set. "
               "Slider ranges are the min/max seen in the data (the model should not be trusted outside them).")
    X = res["X_train"]
    inputs = {}
    cols = st.columns(4)
    helps = {"MedInc": "Median income (tens of thousands of $)", "HouseAge": "Median house age (years)",
             "AveRooms": "Average rooms per household", "AveBedrms": "Average bedrooms per household",
             "Population": "Block-group population", "AveOccup": "Average occupants per household",
             "Latitude": "Latitude", "Longitude": "Longitude"}
    for i, f in enumerate(core.FEATURES):
        lo, hi, med = float(X[f].min()), float(X[f].max()), float(X[f].median())
        step = float(10 ** np.floor(np.log10(max(hi - lo, 1e-6) / 100)))      # sensible step per feature
        inputs[f] = cols[i % 4].number_input(f, min_value=lo, max_value=hi, value=med, step=step,
                                             help=helps[f], format="%.3f")
    which = st.radio("Model", ["Original", "Sensitivity (flagged removed)", "Both"], horizontal=True)
    if st.button("Predict", type="primary"):
        xrow = pd.DataFrame([inputs])[core.FEATURES]
        out = []
        for name, mdl in [("Original", orig), ("Sensitivity", sens)]:
            if which != "Both" and not which.startswith(name):
                continue
            sf = mdl.get_prediction(pd.concat([pd.DataFrame({"const": [1.0]}), xrow], axis=1)).summary_frame(alpha=0.05)
            out.append((name, float(sf["mean"].iloc[0]), float(sf["obs_ci_lower"].iloc[0]), float(sf["obs_ci_upper"].iloc[0])))
        cc = st.columns(len(out))
        for col, (name, p, lo, hi) in zip(cc, out):
            col.metric(f"{name} model", f"${p * 100000:,.0f}", f"95% PI: ${lo * 100000:,.0f} – ${hi * 100000:,.0f}", delta_color="off")
            if p > core.TARGET_CAP:
                col.warning("Above the dataset's cap of $500,000 – treat with caution.")
            if p < 0:
                col.warning("Negative prediction: the linear model is extrapolating badly here.")
        st.caption("PI = 95% prediction interval for a single new block group (assumes roughly normal, "
                   "constant-variance errors – see the diagnostics section for how well that holds).")
