# Predicting 30-day complications and readmissions after surgical discharge

Code for the paper "Predicting 30-Day Complications and Readmissions after Surgical Discharge from Perioperative Records with Patient-Level Validation" (IEEE CISS 2027).

## What the notebook does
`inspire_postdischarge_risk.ipynb` runs the whole analysis:
1. Builds the cohort from INSPIRE and labels each surgery: positive if, within 30 days of discharge, the patient was readmitted, received a new complication code, or died.
2. Builds the features (patient, surgery, 11 preoperative labs, 18 ward vital-sign features) and hourly vital-sign sequences.
3. Splits by patient (StratifiedGroupKFold) into training, validation and test sets.
4. Trains logistic regression, random forest, XGBoost (with and without class weights), an LSTM on vital signs, and runs Llama 3.2 3B zero-shot through Ollama on the local computer.
5. Evaluates ROC-AUC, PR-AUC, Brier score, calibration slope and intercept, with 95% confidence intervals that resample patients; thresholds are chosen on validation data.
6. Validates externally on MIMIC-IV and compares ward vital-sign ranges with WESAD.

## Figures
The `figures_png` folder holds Figs. 1-7 of the paper.

## Data (not included)
No patient data are in this repository. The datasets are available to credentialed users under their data use agreements:
* INSPIRE: https://physionet.org/content/inspire/
* MIMIC-IV: https://physionet.org/content/mimiciv/
* WESAD: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection

Download them and set the data paths at the top of the notebook.

## Requirements
Python 3, pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib, joblib, and Ollama with the `llama3.2:3b` model for the language-model part.
