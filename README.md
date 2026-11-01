# Risk Assessment: Churn Prediction

A logistic regression model that predicts company churn, wrapped in a full ML lifecycle: ingestion, training, scoring, deployment, diagnostics, reporting, a Flask API, and automatic retraining when the score drops.

## What's inside

- `ingestion.py`, `training.py`, `scoring.py` and `deployment.py` are the pipeline steps
- `app.py` is the Flask API
- `fullprocess.py` checks new data and retrains

## How to run

```bash
pip install pandas numpy scikit-learn flask matplotlib seaborn requests
rm ingesteddata/ingestedfiles.txt
python firstRun.py
python fullprocess.py
```
