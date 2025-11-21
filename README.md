# NanoToxicity Prediction

Generalizing a deep learning predictive model trained solely on the physicochemical properties of nanoparticles to perform on an organ-specific toxicity domain.

## Contents
- `nanotox_dataset.csv` — main dataset used for training and exploration.
- `test_data.csv` — test dataset used by both notebooks for evaluation.
- `NanoToxicity_Prediction_ML.ipynb` — Random Forest pipeline: data selection, label encoding, training (RandomForestClassifier), evaluation, and saving model (`RandomForest_NanoToxicity.pkl`).
- `NanoToxicity_Prediction_DL.ipynb` — Deep learning pipeline using TensorFlow/Keras: preprocessing (StandardScaler), MLP definition, training, evaluation, saving model (`nanoparticle_toxicity_model.h5`), prediction helper, PDP and SHAP explainability examples.

## Summary
Both notebooks use the same core input features:
- `coresize`, `hydrosize`, `surfcharge`, `e`, `dosage`

They perform cleaning, label encoding (NonToxic → 0, Toxic → 1), feature scaling for DL, model training, and evaluation on `test_data.csv`.

## Dependencies
Minimal set (approximate — check notebooks for exact imports):

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- tensorflow (for DL notebook)
- shap (optional, for explainability plots)

Quick install (recommended inside a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn joblib tensorflow shap
```

If you prefer to install only what you need for the ML notebook, omit `tensorflow` and `shap`.

## Quick start
1. Ensure `nanotox_dataset.csv` and `test_data.csv` are in the notebook working directory (repo root).
2. Launch Jupyter Lab/Notebook:

```bash
jupyter lab
```

3. Open and run the notebooks in this order (recommended):
   - `NanoToxicity_Prediction_ML.ipynb` — trains and saves a Random Forest model (`RandomForest_NanoToxicity.pkl`).
   - `NanoToxicity_Prediction_DL.ipynb` — trains a Keras MLP, saves `nanoparticle_toxicity_model.h5`, and includes a `predict_toxicity(...)` helper.

4. Use the saved models for quick inference on new samples (notebooks show example prediction calls using the saved models).

## Notes & Tips
- Notebooks use relative paths to `nanotox_dataset.csv` and `test_data.csv`. Run them from the repo root or update the `file_path` paths accordingly.
