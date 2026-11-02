# Regression Diagnostics and Influential Observation Analysis for Real-World Prediction

Statistical Machine Learning project · California Housing · Statsmodels OLS · Streamlit dashboard

## What this project does
Fits a multiple linear regression to predict California median house value, **audits it statistically**
(residual plots, Q-Q, Breusch-Pagan, Jarque-Bera, VIF), finds **outliers / high-leverage / influential observations**
(Cook's D, leverage, studentized residuals, DFFITS), refits the model without the flagged points, and compares both
models on the **same untouched test set**. Everything is shown in an interactive dashboard.

* Target `MedHouseVal` is in **units of $100,000** and is **capped at 5.0** ($500,000).
* Observations are **never deleted automatically**: flagged points are investigated first, and the sensitivity
  analysis asks "does it matter?" rather than assuming removal helps.

## Files
```
Regression_Diagnostics_Project/
├── Regression_Diagnostics_Project.ipynb   # full step-by-step notebook (Modules 1-8)
├── app.py                                 # Streamlit dashboard (7 sections)
├── sml_core.py                            # ALL statistics + plots (shared by notebook, script and dashboard)
├── run_pipeline.py                        # runs everything headless, saves outputs/ and reports/
├── tests/verify_statistics.py             # 26 independent checks (brute-force leave-one-out, leakage, alignment)
├── requirements.txt / requirements-stable.txt
├── data/                                  # CSV cache of the dataset (auto-created)
├── outputs/{diagnostic_plots, influence_analysis, model_comparison}/
├── reports/project_results.md             # real numbers, written by run_pipeline.py
├── reports/academic_material.md           # abstract, slides outline, report structure, viva Q&A
└── .gitignore
```
`sml_core.py` is the single source of truth, so the notebook and the dashboard always give identical results.

## Methodology (data-leakage safe)
1. 80/20 split, `random_state=42`. 2. OLS fitted on **train only**. 3. All diagnostics and influence measures from
the **train** model. 4. Flag: Cook's D > 4/n (default; dashboard also offers DFFITS and "both").
5. Investigate flagged rows, refit without them. 6. Score both models on the identical **test** set.
The test set is never used to fit, flag, or choose a rule.

---
## How to run

### A) Windows laptop (Command Prompt)
Install Python 3.10-3.12 from python.org (tick "Add Python to PATH"), then:
```bat
cd path\to\Regression_Diagnostics_Project
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py          :: runs everything, fills outputs/ and reports/project_results.md
streamlit run app.py            :: opens the dashboard at http://localhost:8501
jupyter notebook                :: then open Regression_Diagnostics_Project.ipynb and "Run All"
```
The first run needs internet (scikit-learn downloads the dataset once and `data/california_housing.csv` is cached).
If `pip install -r requirements.txt` fails, use `pip install -r requirements-stable.txt`.

### B) Google Colab
**Option 1 - Google Drive (recommended)**
1. Upload the whole `Regression_Diagnostics_Project` folder to your Google Drive.
2. Go to colab.research.google.com → File → Open notebook → Google Drive → pick the notebook.
3. Add a first cell and run it:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Regression_Diagnostics_Project
!pip install -q statsmodels plotly
```
4. Runtime → Run all. (Plots and CSVs are saved into `outputs/` in your Drive.)

**Option 2 - upload only the notebook**: upload the `.ipynb` to Colab and Run all; the setup cell will ask you to upload `sml_core.py`.

**Dashboard from Colab (optional)**
```python
!pip install -q streamlit plotly
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501      # open the printed URL; the password is your public IP:  !curl ipv4.icanhazip.com
```

### C) Upload to GitHub
```bat
cd Regression_Diagnostics_Project
git init
git add .
git commit -m "Regression diagnostics SML project"
git branch -M main
git remote add origin https://github.com/<your-username>/Regression_Diagnostics_Project.git
git push -u origin main
```
(Create the empty repository on github.com first - New repository, no README. Run `python run_pipeline.py` *before*
committing so your real outputs/plots and `reports/project_results.md` are included.)

### D) Deploy on Streamlit Community Cloud
1. Push the project to a **public** GitHub repo (section C). `requirements.txt` must be in the repo root.
2. Go to share.streamlit.io → sign in with GitHub → **Create app** → select the repo, branch `main`, main file `app.py`.
3. Advanced settings → choose Python 3.11 or 3.12 → **Deploy**. The first start takes a few minutes
   (it installs the packages and downloads the dataset).

---
## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: statsmodels` | `pip install statsmodels` |
| Version conflicts / import errors | `pip install -r requirements-stable.txt` in a fresh virtual environment |
| `Could not download California Housing` | connect to the internet once; later runs use `data/california_housing.csv` |
| Dashboard opens but is empty | wait for "Loading data and fitting models" to finish (first load only) |
| Colab says `sml_core` not found | use Option 1, or upload `sml_core.py` when asked |

## Step 0 - verify your installation (30 seconds, no internet needed)
```
python tests/verify_statistics.py
```
Expected: `26 / 26 checks passed`. Part A proves the Cook's D / DFFITS / studentized-residual formulas against brute-force
leave-one-out refitting; Part B checks that *your installed statsmodels* and this project's pipeline give the same numbers
as independent calculations, and that there is no train/test leakage or row misalignment. If any line says FAIL, send me
that line before doing anything else.

## Honest note on verification
The project was developed and logic-tested in a sandbox *without internet*, so the real California Housing data and the real
`statsmodels` / `streamlit` / `plotly` packages could not be run there (statsmodels was replaced by a numpy stand-in for the
tests). Therefore **no real results are bundled**: run `python run_pipeline.py` (or the notebook) once and all numbers,
plots and `reports/project_results.md` are produced from your own execution. Read the outputs before your viva - the
interpretation text is generated from the actual values. The Streamlit dashboard has not been rendered in a real browser.
