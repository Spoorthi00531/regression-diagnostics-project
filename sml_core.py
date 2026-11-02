"""
sml_core.py  -  Shared methodology for the whole project
=========================================================
Regression Diagnostics and Influential Observation Analysis (California Housing)

The Jupyter notebook, run_pipeline.py and the Streamlit dashboard (app.py) ALL
import their calculations from this file.  That is how we guarantee that the
notebook and the dashboard produce exactly the same numbers.

Sections
  1. Settings / constants
  2. Data loading and splitting
  3. Regression model (OLS) + metrics
  4. Regression diagnostics (BP, JB, VIF, residual summaries)
  5. Influence analysis (Cook's D, leverage, studentized residuals, DFFITS)
  6. Sensitivity model + comparison
  7. Plain-English interpretation helpers (text is built from the REAL numbers)
  8. Matplotlib plotting helpers (used by the notebook and run_pipeline.py)
  9. Orchestration: run_analysis(), save_all_outputs(), build_results_markdown()
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

# ----------------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------------
RANDOM_STATE = 42          # one seed used everywhere -> reproducible results
TEST_SIZE = 0.20           # 80 % train / 20 % test
TARGET = "MedHouseVal"     # median house value, in units of $100,000
TARGET_CAP = 5.0           # the dataset censors (caps) the target at 5.0
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]
ALPHA = 0.05               # significance level for hypothesis tests
STUD_THRESHOLD = 3.0       # |externally studentized residual| > 3  => outlier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
DIAG_DIR = os.path.join(OUT_DIR, "diagnostic_plots")
INFL_DIR = os.path.join(OUT_DIR, "influence_analysis")
COMP_DIR = os.path.join(OUT_DIR, "model_comparison")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
CSV_CACHE = os.path.join(DATA_DIR, "california_housing.csv")

FLAG_RULES = {
    "cook": "Cook's D > 4/n",
    "dffits": "|DFFITS| > 2*sqrt(p/n)",
    "both": "Cook's D > 4/n AND |DFFITS| > 2*sqrt(p/n)",
}


def ensure_dirs():
    for d in (DATA_DIR, DIAG_DIR, INFL_DIR, COMP_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# 2. DATA
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load California Housing programmatically (no manual download).

    First run needs internet: scikit-learn downloads and caches the data.
    A CSV copy is saved in data/ so later runs also work offline."""
    ensure_dirs()
    try:
        bunch = fetch_california_housing(as_frame=True)
        df = bunch.frame[FEATURES + [TARGET]].copy()
        df.to_csv(CSV_CACHE, index=False)
    except Exception as err:                      # e.g. no internet
        if os.path.exists(CSV_CACHE):
            df = pd.read_csv(CSV_CACHE)
        else:
            raise RuntimeError(
                "Could not download California Housing and no cached copy exists "
                "in data/. Connect to the internet once and re-run.") from err
    return df


def data_overview(df: pd.DataFrame) -> dict:
    """Everything Module 1 asks us to display."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_frame("dtype"),
        "missing": df.isna().sum().to_frame("missing_values"),
        "n_duplicates": int(df.duplicated().sum()),
        "describe": df.describe().T,
        "n_at_cap": int((df[TARGET] >= TARGET_CAP).sum()),
        "pct_at_cap": float((df[TARGET] >= TARGET_CAP).mean() * 100),
    }


def split_data(df: pd.DataFrame):
    """Reproducible train/test split. The TEST set is never used for fitting,
    influence analysis or choosing which observations to flag."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def eda_observations(df: pd.DataFrame) -> list[str]:
    """Statistical observations built from the actual data."""
    obs = []
    t = df[TARGET]
    obs.append(f"Target skewness = {t.skew():.2f}: the distribution is "
               f"{'right-skewed' if t.skew() > 0.5 else 'roughly symmetric'}.")
    n_cap = int((t >= TARGET_CAP).sum())
    obs.append(f"{n_cap} houses ({n_cap / len(df) * 100:.1f}%) sit exactly at the cap of "
               f"{TARGET_CAP} (= $500,000). The true values are unknown (censored), "
               "which will show up as a visible band in the residual plots.")
    corr = df.corr()[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    obs.append(f"Strongest linear correlation with the target: {corr.index[0]} "
               f"(r = {corr.iloc[0]:.2f}); weakest: {corr.index[-1]} (r = {corr.iloc[-1]:.2f}).")
    for col in ["AveRooms", "AveBedrms", "AveOccup", "Population"]:
        ratio = df[col].max() / df[col].median()
        obs.append(f"{col}: max = {df[col].max():,.1f} vs median = {df[col].median():,.2f} "
                   f"(max is {ratio:,.0f}x the median)"
                   + (" -> extreme values present, candidates for high leverage." if ratio > 10 else "."))
    fc = df[FEATURES].corr().abs().where(~np.eye(len(FEATURES), dtype=bool)).stack()
    a, b = fc.idxmax()
    obs.append(f"Most correlated feature pair: {a} & {b} (|r| = {fc.max():.2f}) -> "
               "watch for multicollinearity (checked later with VIF).")
    return obs


# ----------------------------------------------------------------------------
# 3. MODEL
# ----------------------------------------------------------------------------
def fit_ols(X: pd.DataFrame, y: pd.Series):
    """Multiple linear regression with statsmodels OLS (adds the intercept)."""
    return sm.OLS(y, sm.add_constant(X)).fit()


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted OLS model.  has_constant='add' makes sure the
    intercept column is added even if X has only ONE row."""
    return np.asarray(model.predict(sm.add_constant(X, has_constant="add")))


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse))}


def coefficient_table(model) -> pd.DataFrame:
    ci = model.conf_int()
    return pd.DataFrame({"coef": model.params, "std_err": model.bse,
                         "t": model.tvalues, "p_value": model.pvalues,
                         "ci_low": ci[0], "ci_high": ci[1]})


# ----------------------------------------------------------------------------
# 4. DIAGNOSTICS
# ----------------------------------------------------------------------------
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    exog = sm.add_constant(X)
    rows = [(c, variance_inflation_factor(exog.values, i))
            for i, c in enumerate(exog.columns) if c != "const"]
    return pd.DataFrame(rows, columns=["Feature", "VIF"]).sort_values("VIF", ascending=False).reset_index(drop=True)


def run_diagnostics(model, X_train: pd.DataFrame) -> dict:
    """Breusch-Pagan, Jarque-Bera, VIF and basic residual summaries (training data)."""
    resid = np.asarray(model.resid)
    lm, lm_p, f, f_p = het_breuschpagan(resid, model.model.exog)
    jb, jb_p, skew, kurt = jarque_bera(resid)
    return {
        "bp": {"LM": lm, "LM_p": lm_p, "F": f, "F_p": f_p},
        "jb": {"JB": jb, "p": jb_p, "skew": skew, "kurtosis": kurt},
        "vif": compute_vif(X_train),
        "resid_mean": float(resid.mean()), "resid_std": float(resid.std(ddof=1)),
        "durbin_watson": float(durbin_watson(resid)),
    }


# ----------------------------------------------------------------------------
# 5. INFLUENCE
# ----------------------------------------------------------------------------
def influence_thresholds(model) -> dict:
    n = int(model.nobs)
    p = int(model.df_model) + 1                 # number of parameters incl. intercept
    return {"n": n, "p": p,
            "cook": 4.0 / n,                    # common rule of thumb
            "leverage": 2.0 * p / n,            # 2p/n  (twice the average leverage p/n)
            "studentized": STUD_THRESHOLD,      # |t_i| > 3
            "dffits": 2.0 * np.sqrt(p / n)}     # 2*sqrt(p/n)


def influence_table(model, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """One row per TRAINING observation with all influence measures."""
    infl = model.get_influence()
    th = influence_thresholds(model)
    df = pd.DataFrame({
        "Obs_ID": X_train.index.to_numpy(),     # row id in the original dataset
        "Cook_D": infl.cooks_distance[0],
        "Leverage": infl.hat_matrix_diag,
        "Studentized_Resid": infl.resid_studentized_external,
        "DFFITS": infl.dffits[0],
        "Actual": y_train.to_numpy(),
        "Fitted": np.asarray(model.fittedvalues),
        "Residual": np.asarray(model.resid),
    })
    df = pd.concat([df, X_train.reset_index(drop=True)], axis=1)
    df["Flag_Cook"] = df["Cook_D"] > th["cook"]
    df["Flag_Leverage"] = df["Leverage"] > th["leverage"]
    df["Flag_Outlier"] = df["Studentized_Resid"].abs() > th["studentized"]
    df["Flag_DFFITS"] = df["DFFITS"].abs() > th["dffits"]
    df["N_Criteria"] = df[["Flag_Cook", "Flag_Leverage", "Flag_Outlier", "Flag_DFFITS"]].sum(axis=1)
    out, lev = df["Flag_Outlier"], df["Flag_Leverage"]
    other = df["Flag_Cook"] | df["Flag_DFFITS"]
    df["Type"] = np.select([out & lev, out & ~lev, ~out & lev, other],
                           ["Outlier + High leverage", "Outlier only (large residual)", "High leverage only",
                            "Cook/DFFITS only (not an outlier or high-leverage by rule)"],
                           default="Not flagged")
    df["Cook_Rank"] = df["Cook_D"].rank(ascending=False, method="first").astype(int)
    return df.sort_values("Cook_Rank").reset_index(drop=True)


def flag_observations(infl_df: pd.DataFrame, rule: str = "cook") -> pd.Series:
    """Boolean mask of POTENTIALLY influential observations (a flag is not a verdict)."""
    if rule == "cook":
        return infl_df["Flag_Cook"].copy()
    if rule == "dffits":
        return infl_df["Flag_DFFITS"].copy()
    if rule == "both":
        return infl_df["Flag_Cook"] & infl_df["Flag_DFFITS"]
    raise ValueError(f"rule must be one of {list(FLAG_RULES)}")


def investigate_flagged(infl_df: pd.DataFrame, flagged: pd.Series):
    """Investigate (not just delete!) flagged observations.
    Returns (profile table, type-count table)."""
    cols = FEATURES + [TARGET]
    tmp = infl_df.copy()
    tmp[TARGET] = tmp["Actual"]            # the target is stored as 'Actual' in the table
    prof = pd.DataFrame({"Flagged_mean": tmp.loc[flagged, cols].mean(),
                         "Not_flagged_mean": tmp.loc[~flagged, cols].mean(),
                         "Flagged_median": tmp.loc[flagged, cols].median(),
                         "Not_flagged_median": tmp.loc[~flagged, cols].median()})
    prof.loc["% at target cap (5.0)"] = [
        (tmp.loc[flagged, "Actual"] >= TARGET_CAP).mean() * 100,
        (tmp.loc[~flagged, "Actual"] >= TARGET_CAP).mean() * 100, np.nan, np.nan]
    types = tmp.loc[flagged, "Type"].value_counts().rename_axis("Type").reset_index(name="Count")
    types["Percent"] = types["Count"] / max(int(flagged.sum()), 1) * 100
    return prof, types


# ----------------------------------------------------------------------------
# 6. SENSITIVITY + COMPARISON
# ----------------------------------------------------------------------------
def fit_sensitivity(X_train, y_train, infl_df: pd.DataFrame, flagged: pd.Series):
    """Refit OLS WITHOUT the flagged training observations.
    infl_df is sorted by Cook rank, so rows are matched by Obs_ID (not by position)."""
    drop_ids = infl_df.loc[flagged, "Obs_ID"].to_numpy()
    keep = ~X_train.index.isin(drop_ids)
    return fit_ols(X_train[keep], y_train[keep]), keep


def compare_models(orig, sens, X_test, y_test, X_train, y_train, keep_mask) -> dict:
    """Evaluate BOTH models on the exact same untouched test set."""
    p_o, p_s = predict(orig, X_test), predict(sens, X_test)
    mo, ms = regression_metrics(y_test, p_o), regression_metrics(y_test, p_s)
    table = pd.DataFrame({"Original": mo, "Sensitivity": ms})
    table["Change"] = table["Sensitivity"] - table["Original"]
    table["Pct_Change"] = 100 * table["Change"] / table["Original"].abs()
    # training metrics (each model on the data it was fitted on) - for information only
    tr = pd.DataFrame({
        "Original (all train)": regression_metrics(y_train, np.asarray(orig.fittedvalues)),
        "Sensitivity (flagged removed)": regression_metrics(y_train[keep_mask], np.asarray(sens.fittedvalues))})
    # coefficient comparison
    ci = orig.conf_int()
    coef = pd.DataFrame({"Original": orig.params, "Sensitivity": sens.params})
    coef["Abs_Change"] = coef["Sensitivity"] - coef["Original"]
    coef["Pct_Change"] = 100 * coef["Abs_Change"] / coef["Original"].abs()
    # scale-free shift: how many ORIGINAL standard errors did the coefficient move?
    # (more reliable than % change, which explodes for coefficients near zero)
    coef["Shift_in_SE"] = coef["Abs_Change"] / orig.bse
    coef["Orig_CI_Low"], coef["Orig_CI_High"] = ci[0], ci[1]
    coef["Sens_Inside_Orig_CI"] = (coef["Sensitivity"] >= ci[0]) & (coef["Sensitivity"] <= ci[1])
    coef["Sign_Flip"] = np.sign(coef["Original"]) != np.sign(coef["Sensitivity"])
    coef["Orig_p"], coef["Sens_p"] = orig.pvalues, sens.pvalues
    # paired test on absolute test errors: is the accuracy difference statistically real?
    ae_o, ae_s = np.abs(np.asarray(y_test) - p_o), np.abs(np.asarray(y_test) - p_s)
    t_stat, t_p = stats.ttest_rel(ae_o, ae_s)
    return {"test_metrics": table, "train_metrics": tr, "coef": coef,
            "pred_orig": p_o, "pred_sens": p_s,
            "paired_t": {"t": float(t_stat), "p": float(t_p),
                         "mean_abs_err_orig": float(ae_o.mean()), "mean_abs_err_sens": float(ae_s.mean())}}


def threshold_sweep(X_train, y_train, X_test, y_test, infl_df) -> pd.DataFrame:
    """How sensitive are the conclusions to the flagging rule?
    NOTE: test metrics are only REPORTED here; they are never used to pick a rule."""
    rules = {
        "Cook's D > 4/n": infl_df["Flag_Cook"],
        "Cook's D > 4/n AND |DFFITS| > 2*sqrt(p/n)": infl_df["Flag_Cook"] & infl_df["Flag_DFFITS"],
        "|Studentized resid| > 3": infl_df["Flag_Outlier"],
        "Top 2% by Cook's D": infl_df["Cook_D"] >= infl_df["Cook_D"].quantile(0.98),
        "Top 1% by Cook's D": infl_df["Cook_D"] >= infl_df["Cook_D"].quantile(0.99),
        "Top 0.5% by Cook's D": infl_df["Cook_D"] >= infl_df["Cook_D"].quantile(0.995),
    }
    base = fit_ols(X_train, y_train)
    base_m = regression_metrics(y_test, predict(base, X_test))
    rows = []
    for name, mask in rules.items():
        keep = ~X_train.index.isin(infl_df.loc[mask, "Obs_ID"].to_numpy())
        m = fit_ols(X_train[keep], y_train[keep])
        met = regression_metrics(y_test, predict(m, X_test))
        shift = (m.params[FEATURES] - base.params[FEATURES]) / base.bse[FEATURES]
        rows.append({"Rule": name, "N_removed": int((~keep).sum()),
                     "Pct_removed": (~keep).mean() * 100,
                     "Test_R2": met["R2"], "Test_RMSE": met["RMSE"],
                     "dRMSE_%_vs_original": 100 * (met["RMSE"] - base_m["RMSE"]) / base_m["RMSE"],
                     "Max_coef_shift_in_SE": float(shift.abs().max()),
                     "Most_shifted_coef": shift.abs().idxmax()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 7. INTERPRETATION TEXT (always generated from the real numbers)
# ----------------------------------------------------------------------------
def interpret_model(model, test_m: dict) -> str:
    sig = [c for c in FEATURES if model.pvalues[c] < ALPHA]
    top = model.params[FEATURES].abs().idxmax()
    return (f"Training R2 = {model.rsquared:.4f} and adjusted R2 = {model.rsquared_adj:.4f}: the model "
            f"explains about {model.rsquared * 100:.1f}% of the variance of house values in the training data. "
            f"The test R2 is {test_m['R2']:.4f} and the test RMSE is {test_m['RMSE']:.4f} "
            f"(about ${test_m['RMSE'] * 100000:,.0f}). "
            f"{len(sig)} of {len(FEATURES)} features are significant at the {ALPHA:.0%} level "
            f"({', '.join(sig) if sig else 'none'}). The coefficient with the largest raw size is {top} "
            f"({model.params[top]:+.4f}); remember raw sizes depend on each feature's units, so they are not a "
            "ranking of importance. Because the sample is large, even tiny effects get small p-values - "
            "always judge practical size too.")


def interpret_bp(d: dict) -> str:
    p = d["bp"]["LM_p"]
    if p < ALPHA:
        return (f"Breusch-Pagan LM = {d['bp']['LM']:.2f}, p = {p:.3g} < {ALPHA}: we REJECT constant variance. "
                "Heteroscedasticity is present, so the usual OLS standard errors/p-values are unreliable "
                "(coefficients stay unbiased). Robust (HC3) standard errors or a transformation are advisable.")
    return (f"Breusch-Pagan LM = {d['bp']['LM']:.2f}, p = {p:.3g} >= {ALPHA}: no significant evidence "
            "against constant variance.")


def interpret_jb(d: dict) -> str:
    j = d["jb"]
    base = (f"Jarque-Bera = {j['JB']:.1f}, p = {j['p']:.3g}; residual skewness = {j['skew']:.2f}, "
            f"kurtosis = {j['kurtosis']:.2f} (normal = 0 and 3). ")
    if j["p"] < ALPHA:
        return base + ("We REJECT normality of residuals. With a large sample the test is very sensitive, so also "
                       "look at the Q-Q plot to judge how serious the departure is. Predictions remain unbiased, but "
                       "t-tests / prediction intervals that assume normal errors are only approximate.")
    return base + "No significant evidence against normality."


def interpret_vif(vif: pd.DataFrame) -> str:
    high = vif[vif["VIF"] > 5]
    sev = vif[vif["VIF"] > 10]
    if high.empty:
        return f"All VIFs <= 5 (max = {vif['VIF'].max():.2f}): multicollinearity is not a concern."
    return (f"{len(high)} feature(s) have VIF > 5 ({', '.join(f'{r.Feature}={r.VIF:.1f}' for r in high.itertuples())}); "
            f"{len(sev)} exceed 10. These variables overlap heavily with others, so their individual coefficients "
            "are unstable (large standard errors) even though overall predictions can still be fine.")


def interpret_influence(infl: pd.DataFrame, th: dict, flagged: pd.Series, rule: str) -> str:
    n = len(infl)
    return (f"Out of {n:,} training observations: {int(infl.Flag_Cook.sum()):,} exceed Cook's D > 4/n "
            f"({th['cook']:.5f}); {int(infl.Flag_Leverage.sum()):,} have leverage > 2p/n ({th['leverage']:.5f}); "
            f"{int(infl.Flag_Outlier.sum()):,} have |studentized residual| > {th['studentized']:.0f}; "
            f"{int(infl.Flag_DFFITS.sum()):,} have |DFFITS| > {th['dffits']:.4f}. The maximum Cook's D is "
            f"{infl.Cook_D.max():.4f} (observation {int(infl.loc[0, 'Obs_ID'])}); a value below 1 means no single "
            f"point dominates the fit, although many small influences can still add up. "
            f"Flagging rule used for the sensitivity analysis: {FLAG_RULES[rule]} -> {int(flagged.sum()):,} "
            f"observations ({flagged.mean() * 100:.2f}%) flagged. These are POTENTIALLY influential - flagging is a "
            "reason to investigate, not to delete.")


def interpret_comparison(comp: dict, n_removed: int) -> str:
    """Two separate verdicts: (a) predictive accuracy, (b) coefficient stability."""
    tm, coef = comp["test_metrics"], comp["coef"].drop(index="const", errors="ignore")
    d_r2, d_rmse = tm.loc["R2", "Change"], tm.loc["RMSE", "Pct_Change"]
    worst = coef["Shift_in_SE"].abs().idxmax()
    worst_se = coef.loc[worst, "Shift_in_SE"]
    n_out_ci = int((~coef["Sens_Inside_Orig_CI"]).sum())
    n_flip = int(coef["Sign_Flip"].sum())
    flipped = ", ".join(coef.index[coef["Sign_Flip"]]) or "none"
    # (a) predictive accuracy on the untouched test set
    pred = ("negligible" if abs(d_rmse) < 1 else "modest" if abs(d_rmse) < 5 else "substantial")
    # (b) coefficient stability, measured in original standard errors
    coefv = ("stable (all shifts < 1 SE)" if abs(worst_se) < 1 else
             "moderately shifted (all inside the original 95% CI)" if n_out_ci == 0 else
             f"clearly shifted ({n_out_ci} coefficient(s) left the original 95% CI)")
    direction = "lower (better)" if d_rmse < 0 else "higher (worse)" if d_rmse > 0 else "unchanged"
    p = comp["paired_t"]["p"]
    return (f"After removing {n_removed:,} flagged observations, the test RMSE changed by {d_rmse:+.2f}% "
            f"({direction}) and the test R2 by {d_r2:+.4f} -> impact on PREDICTIVE ACCURACY is {pred} "
            f"(paired t-test on absolute test errors: p = {p:.3g}, "
            f"{'statistically detectable' if p < ALPHA else 'not statistically detectable'}). "
            f"COEFFICIENTS are {coefv}; the largest move is {worst} ({worst_se:+.2f} original standard errors); "
            f"sign flips: {flipped}. "
            "Heuristics used: RMSE change <1% negligible, <5% modest; coefficient shift <1 SE stable, "
            "outside the 95% CI = clearly shifted. Removing observations does not automatically improve accuracy: "
            "the test set still contains similar extreme points that the sensitivity model never saw. "
            "Sign flips of coefficients near zero are not alarming - check Shift_in_SE first.")


# ----------------------------------------------------------------------------
# 8. MATPLOTLIB PLOTS
# ----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")


def _finish(fig, save_path=None):
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def plot_target_dist(df, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df[TARGET], bins=50, kde=True, ax=ax, color="#2b6cb0")
    ax.axvline(TARGET_CAP, color="crimson", ls="--", label=f"Cap = {TARGET_CAP} ($500k)")
    ax.set(title="Distribution of MedHouseVal (units of $100,000)", xlabel="MedHouseVal")
    ax.legend()
    return _finish(fig, save_path)


def plot_feature_dists(df, save_path=None):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.histplot(df[col], bins=40, ax=ax, color="#38a169")
        ax.set_title(col)
    for ax in axes.ravel()[len(FEATURES):]:
        ax.axis("off")
    return _finish(fig, save_path)


def plot_corr(df, save_path=None):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    return _finish(fig, save_path)


def plot_feature_vs_target(df, save_path=None, sample=3000):
    s = df.sample(min(sample, len(df)), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.regplot(x=col, y=TARGET, data=s, ax=ax, scatter_kws={"s": 6, "alpha": 0.3},
                    line_kws={"color": "crimson"})
        ax.set_title(f"{col} vs {TARGET}")
    fig.suptitle(f"Feature vs target (random sample of {len(s):,} rows)", y=1.01)
    return _finish(fig, save_path)


def plot_actual_vs_pred(y_true, y_pred, title, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3)
    lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set(xlabel="Actual", ylabel="Predicted", title=title)
    ax.legend()
    return _finish(fig, save_path)


def plot_resid_vs_fitted(fitted, resid, save_path=None):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    fitted, resid = np.asarray(fitted), np.asarray(resid)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted, resid, s=5, alpha=0.25)
    ax.axhline(0, color="red", ls="--")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(fitted), min(3000, len(fitted)), replace=False)
    sm_line = lowess(resid[idx], fitted[idx], frac=0.3)
    ax.plot(sm_line[:, 0], sm_line[:, 1], color="orange", lw=2, label="LOWESS trend")
    ax.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted values")
    ax.legend()
    return _finish(fig, save_path)


def plot_resid_hist(resid, save_path=None):
    resid = np.asarray(resid)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(resid, bins=60, stat="density", ax=ax, color="#805ad5")
    x = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(x, stats.norm.pdf(x, resid.mean(), resid.std(ddof=1)), "r-", label="Normal curve")
    ax.set(title="Residual distribution", xlabel="Residual")
    ax.legend()
    return _finish(fig, save_path)


def plot_qq(resid, save_path=None):
    (osm, osr), (slope, intercept, _) = stats.probplot(np.asarray(resid), dist="norm")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(osm, osr, s=5, alpha=0.4)
    ax.plot(osm, slope * osm + intercept, "r-", label="Reference line")
    ax.set(title="Normal Q-Q plot of residuals", xlabel="Theoretical quantiles", ylabel="Sample quantiles")
    ax.legend()
    return _finish(fig, save_path)


def plot_vif(vif, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=vif, x="VIF", y="Feature", ax=ax, color="#dd6b20")
    ax.axvline(5, color="gold", ls="--", label="VIF = 5")
    ax.axvline(10, color="red", ls="--", label="VIF = 10")
    ax.set_title("Variance Inflation Factors")
    ax.legend()
    return _finish(fig, save_path)


def plot_cooks(infl, th, save_path=None):
    d = infl.sort_values("Obs_ID").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = np.where(d["Flag_Cook"], "crimson", "steelblue")
    ax.vlines(d.index, 0, d["Cook_D"], colors=colors, lw=0.6)
    ax.axhline(th["cook"], color="orange", ls="--", label=f"4/n = {th['cook']:.5f}")
    for _, r in infl.head(5).iterrows():
        ax.annotate(str(int(r["Obs_ID"])), (d.index[d["Obs_ID"] == r["Obs_ID"]][0], r["Cook_D"]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set(title="Cook's distance (red = above 4/n; labels = top-5 Obs_ID)",
           xlabel="Training observation (ordered by dataset row id)", ylabel="Cook's D")
    ax.legend()
    return _finish(fig, save_path)


def plot_leverage_resid(infl, th, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    flagged = infl["Flag_Cook"]
    ax.scatter(infl.loc[~flagged, "Leverage"], infl.loc[~flagged, "Studentized_Resid"],
               s=6, alpha=0.3, color="steelblue", label="Not flagged by Cook")
    ax.scatter(infl.loc[flagged, "Leverage"], infl.loc[flagged, "Studentized_Resid"],
               s=10 + 3000 * infl.loc[flagged, "Cook_D"].clip(upper=0.1), alpha=0.7, color="crimson",
               label="Cook's D > 4/n (size ~ Cook's D)")
    ax.axvline(th["leverage"], color="green", ls="--", label=f"Leverage 2p/n = {th['leverage']:.4f}")
    for v in (-th["studentized"], th["studentized"]):
        ax.axhline(v, color="purple", ls="--", lw=1)
    ax.set(title="Leverage vs externally studentized residual", xlabel="Leverage (hat value)",
           ylabel="Studentized residual")
    ax.legend(fontsize=8)
    return _finish(fig, save_path)


def plot_metrics_compare(test_metrics, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, m in zip(axes, ["R2", "MAE", "MSE", "RMSE"]):
        vals = [test_metrics.loc[m, "Original"], test_metrics.loc[m, "Sensitivity"]]
        ax.bar(["Original", "Sensitivity"], vals, color=["#2b6cb0", "#dd6b20"])
        ax.set_title(f"Test {m}")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    return _finish(fig, save_path)


def plot_coef_compare(coef, save_path=None):
    c = coef.drop(index="const", errors="ignore")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["seagreen" if ok else "crimson" for ok in c["Sens_Inside_Orig_CI"]]
    ax.barh(c.index, c["Shift_in_SE"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    for v in (-1.96, 1.96):
        ax.axvline(v, color="gray", ls="--", lw=1)
    ax.set(title="Coefficient shift after removing flagged points (in original standard errors)\n"
                 "green = inside original 95% CI, red = outside; dashed = +/-1.96 SE",
           xlabel="Shift / original SE")
    return _finish(fig, save_path)


# ----------------------------------------------------------------------------
# 9. ORCHESTRATION
# ----------------------------------------------------------------------------
def run_analysis(rule: str = "cook", df: pd.DataFrame | None = None) -> dict:
    """Run the entire methodology once and return every result in a dict."""
    df = load_data() if df is None else df
    X_train, X_test, y_train, y_test = split_data(df)

    orig = fit_ols(X_train, y_train)                         # fitted on TRAIN only
    test_pred = predict(orig, X_test)
    metrics = {"train": regression_metrics(y_train, np.asarray(orig.fittedvalues)),
               "test": regression_metrics(y_test, test_pred)}
    diag = run_diagnostics(orig, X_train)

    infl = influence_table(orig, X_train, y_train)           # TRAIN only (no leakage)
    th = influence_thresholds(orig)
    infl.attrs["thresholds"] = th
    flagged = flag_observations(infl, rule)
    profile, type_counts = investigate_flagged(infl, flagged)

    sens, keep_mask = fit_sensitivity(X_train, y_train, infl, flagged)
    comp = compare_models(orig, sens, X_test, y_test, X_train, y_train, keep_mask)
    sweep = threshold_sweep(X_train, y_train, X_test, y_test, infl)

    return {
        "df": df, "overview": data_overview(df), "eda_notes": eda_observations(df),
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "orig": orig, "sens": sens, "metrics": metrics, "test_pred": test_pred,
        "diag": diag, "infl": infl, "thresholds": th, "rule": rule,
        "flagged": flagged, "keep_mask": keep_mask, "profile": profile, "type_counts": type_counts,
        "comp": comp, "sweep": sweep,
        "text": {
            "model": interpret_model(orig, metrics["test"]),
            "bp": interpret_bp(diag), "jb": interpret_jb(diag), "vif": interpret_vif(diag["vif"]),
            "influence": interpret_influence(infl, th, flagged, rule),
            "comparison": interpret_comparison(comp, int(flagged.sum())),
        },
    }


def save_all_outputs(res: dict) -> list[str]:
    """Save every plot and CSV into outputs/ (used by run_pipeline.py)."""
    ensure_dirs()
    df, orig, infl, th, comp = res["df"], res["orig"], res["infl"], res["thresholds"], res["comp"]
    d, i, c = DIAG_DIR, INFL_DIR, COMP_DIR

    def _save(fn, *args):               # draw, save to disk, then free the figure
        plt.close(fn(*args))

    _save(plot_target_dist, df, f"{d}/01_target_distribution.png")
    _save(plot_feature_dists, df, f"{d}/02_feature_distributions.png")
    _save(plot_corr, df, f"{d}/03_correlation_heatmap.png")
    _save(plot_feature_vs_target, df, f"{d}/04_feature_vs_target.png")
    _save(plot_actual_vs_pred, res["y_test"], res["test_pred"], "Actual vs predicted (TEST set)", f"{d}/05_actual_vs_predicted_test.png")
    _save(plot_resid_vs_fitted, orig.fittedvalues, orig.resid, f"{d}/06_residuals_vs_fitted.png")
    _save(plot_resid_hist, orig.resid, f"{d}/07_residual_histogram.png")
    _save(plot_qq, orig.resid, f"{d}/08_qq_plot.png")
    _save(plot_vif, res["diag"]["vif"], f"{d}/09_vif.png")
    _save(plot_cooks, infl, th, f"{i}/cooks_distance.png")
    _save(plot_leverage_resid, infl, th, f"{i}/leverage_vs_studentized_residual.png")
    _save(plot_metrics_compare, comp["test_metrics"], f"{c}/test_metrics_comparison.png")
    _save(plot_coef_compare, comp["coef"], f"{c}/coefficient_changes.png")
    plt.close("all")

    infl.to_csv(f"{i}/influence_table_all_training_obs.csv", index=False)
    infl[res["flagged"]].to_csv(f"{i}/flagged_observations.csv", index=False)
    infl.head(50).to_csv(f"{i}/top50_influential.csv", index=False)
    res["profile"].to_csv(f"{i}/flagged_vs_rest_profile.csv")
    comp["test_metrics"].to_csv(f"{c}/test_metrics_comparison.csv")
    comp["train_metrics"].to_csv(f"{c}/train_metrics_comparison.csv")
    comp["coef"].to_csv(f"{c}/coefficient_comparison.csv")
    res["sweep"].to_csv(f"{c}/threshold_sensitivity.csv", index=False)
    res["diag"]["vif"].to_csv(f"{d}/vif.csv", index=False)
    coefficient_table(orig).to_csv(f"{d}/ols_coefficients.csv")
    return sorted(os.listdir(d)) + sorted(os.listdir(i)) + sorted(os.listdir(c))


def _md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    """Tiny DataFrame -> markdown table (avoids needing the 'tabulate' package)."""
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    rows = ""
    for _, r in d.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        rows += "| " + " | ".join(cells) + " |\n"
    return head + rows


def build_results_markdown(res: dict) -> str:
    """Module 8: a results document where EVERY number comes from the code run."""
    ov, m, diag, th = res["overview"], res["metrics"], res["diag"], res["thresholds"]
    comp, orig, flagged = res["comp"], res["orig"], res["flagged"]
    top10 = res["infl"].head(10)[["Cook_Rank", "Obs_ID", "Cook_D", "Leverage", "Studentized_Resid",
                                  "DFFITS", "Actual", "Fitted", "Type"]]
    tm = comp["test_metrics"]
    s = f"""# Project Results (auto-generated from the executed code)

> Generated by `run_pipeline.py`. Re-running it with the same library versions reproduces every number
> (random seed = {RANDOM_STATE}, test size = {TEST_SIZE}).

## 1. Data
- Rows x columns: **{ov['shape'][0]:,} x {ov['shape'][1]}**; missing values: **{int(ov['missing'].sum().iloc[0])}**; duplicate rows: **{ov['n_duplicates']}**
- Target `{TARGET}` is in units of **$100,000** and is **capped at {TARGET_CAP}**; {ov['n_at_cap']:,} rows ({ov['pct_at_cap']:.2f}%) sit at the cap.
- Train / test sizes: **{len(res['X_train']):,} / {len(res['X_test']):,}**

### EDA observations
""" + "\n".join(f"- {o}" for o in res["eda_notes"]) + f"""

## 2. OLS model (fitted on training data only)
| Split | R2 | MAE | MSE | RMSE |
|---|---|---|---|---|
| Train | {m['train']['R2']:.4f} | {m['train']['MAE']:.4f} | {m['train']['MSE']:.4f} | {m['train']['RMSE']:.4f} |
| Test | {m['test']['R2']:.4f} | {m['test']['MAE']:.4f} | {m['test']['MSE']:.4f} | {m['test']['RMSE']:.4f} |

Adjusted R2 (train) = {orig.rsquared_adj:.4f}.

{res['text']['model']}

### Coefficients
{_md(coefficient_table(orig))}

<details><summary>Full statsmodels summary</summary>

```
{orig.summary().as_text()}
```
</details>

## 3. Regression diagnostics
- **Breusch-Pagan:** {res['text']['bp']}
- **Jarque-Bera:** {res['text']['jb']}
- **VIF:** {res['text']['vif']}
- Durbin-Watson = {diag['durbin_watson']:.3f} (about 2 means no first-order autocorrelation; the data has no time order so this is only informational).

{_md(diag['vif'])}

## 4. Influential observation analysis
Thresholds: n = {th['n']:,}, p = {th['p']}; Cook's D > {th['cook']:.6f}; leverage > {th['leverage']:.6f};
|studentized residual| > {th['studentized']:.0f}; |DFFITS| > {th['dffits']:.4f}.

{res['text']['influence']}

### Top 10 observations by Cook's distance
{_md(top10)}

### What are the flagged observations?
{_md(res['type_counts'])}

{_md(res['profile'])}

## 5. Original vs sensitivity model (same untouched test set)
{_md(tm)}

{res['text']['comparison']}

### Coefficient comparison
{_md(comp['coef'][['Original', 'Sensitivity', 'Pct_Change', 'Shift_in_SE', 'Sens_Inside_Orig_CI', 'Sign_Flip']])}

### Does the conclusion depend on the flagging rule?
{_md(res['sweep'])}

## 6. Key findings
- {res['text']['model'].split('. ')[0]}.
- Heteroscedasticity: {'present' if diag['bp']['LM_p'] < ALPHA else 'not detected'} (BP p = {diag['bp']['LM_p']:.3g}); residual normality: {'rejected' if diag['jb']['p'] < ALPHA else 'not rejected'} (JB p = {diag['jb']['p']:.3g}).
- Largest VIF = {diag['vif'].iloc[0]['Feature']} ({diag['vif'].iloc[0]['VIF']:.1f}).
- {int(flagged.sum()):,} training observations ({flagged.mean() * 100:.2f}%) were flagged as potentially influential; test RMSE changed by {tm.loc['RMSE', 'Pct_Change']:+.2f}% when they were excluded.

## 7. Limitations
- Linear model: cannot capture the non-linear location effect (latitude/longitude) or interactions.
- Target censored at {TARGET_CAP}: OLS treats capped values as real values, which biases fits near the top.
- Block-group level averages, not individual houses; 1990 census data (prices are not current).
- Thresholds (4/n, 2p/n, |t|>3, 2*sqrt(p/n)) are conventions, not laws; with large n they flag many points.
- Influence measures were computed on training data only; masking effects (several similar outliers hiding each other) are possible.

## 8. Conclusion
{res['text']['comparison']}

## 9. Future enhancements
Robust regression (Huber / RANSAC), heteroscedasticity-consistent (HC3) standard errors, log-transform of the target,
spatial features or tree-based models, handling the censored target (Tobit), cross-validated influence analysis,
and DFBETAS per coefficient.
"""
    return s
