# Risk Assessment: Churn Prediction with Automated Retraining

A logistic regression model that predicts whether a company will churn (`exited`) from three
features: `lastmonth_activity`, `lastyear_activity`, and `number_of_employees`. Around the model,
this repo builds the full lifecycle of a production ML system: data ingestion, training, scoring,
deployment, diagnostics, reporting, and a Flask API, plus an automated loop that detects model
drift on new data and retrains when the score drops.

The datasets are small fabricated company records, so the repo is best read as a working template
for ML model operations rather than a production model.

## Repository structure

- `ingestion.py` - Merges source CSVs into one deduplicated dataset and records which files were
  ingested so repeated runs only pick up new data.
- `training.py` - Trains the logistic regression and saves `trainedmodel.sav` plus `scaler.sav`.
- `data_prep.py` - Shared feature extraction and scaling used by training, scoring, reporting,
  diagnostics, and the API.
- `scoring.py` - Scores the trained model on the test data (F1 score) and writes
  `latestscore.txt`.
- `deployment.py` - Copies the model, scaler, score, and ingest record into
  `production_deployment/`.
- `diagnostics.py` - Model predictions, dataset summary statistics, ingestion/training timings,
  and a check for outdated Python packages. Writes `req_versions.csv`.
- `reporting.py` - Plots and saves a confusion matrix for the model on the test data.
- `app.py` - Flask API with endpoints for predictions, scoring, summary stats, and diagnostics.
- `wsgi.py` - WSGI entry point for serving the API with a production server such as gunicorn.
- `apicalls.py` - Calls every API endpoint and saves the combined responses to
  `apireturns.txt`.
- `firstRun.py` - Runs the whole pipeline once on the practice data (ingest, train, score,
  deploy, diagnostics, report).
- `fullprocess.py` - The automated loop: checks for new source data, ingests it, scores the
  deployed model on it, and retrains plus redeploys when the score drops. Meant to run on a
  schedule (see `cron.txt` / `cronjob.txt`).
- `eda.ipynb` - Notebook that scores the model and exercises the live API.
- `config.json` - Path configuration (input data, ingested data, test data, models, deployment).
- `requirements.txt` - Pinned Python dependencies (the canonical dependency list).

Data and artifact directories: `sourcedata/`, `practicedata/`, `ingesteddata/`, `testdata/`,
`models/`, `practicemodels/`, `production_deployment/`.

Note on `requirements_installed.txt`: this file is not a second dependency list. It is the
output snapshot that `diagnostics.py` regenerates from `pip list --outdated` every time the
outdated-packages check runs. Treat it as a report, not as something to install from.

## How to run it

Install dependencies (Python 3.8+ recommended, matching the pinned versions):

```
pip install -r requirements.txt
```

Run the full pipeline once on the practice data:

```
python firstRun.py
```

Start the API (from the repo root, in a separate terminal):

```
python app.py
```

Then check the endpoints, for example:

```
curl "http://127.0.0.1:8000/scoring"
curl "http://127.0.0.1:8000/prediction?filename=finaldata2.csv"
```

Exercise all endpoints and save the responses:

```
python apicalls.py
```

Run the automated retraining loop (checks for new data in `sourcedata/` and retrains on drift):

```
python fullprocess.py
```

Open `eda.ipynb` for a notebook walkthrough of scoring and the API.

## What you learn from this repo

- How to structure an end-to-end ML pipeline so each stage (ingest, train, score, deploy)
  is a separate, rerunnable script driven by a single `config.json`.
- How to persist and reuse a feature scaler so training and serving apply identical
  transformations.
- How to watch a deployed model for drift: score it on newly arrived data, compare with the
  recorded score, and retrain automatically when performance drops.
- How to expose model health through a small API: predictions, scores, data summaries, and
  timing/dependency diagnostics in one place.

## Notes

- `firstRun.py` and `fullprocess.py` rewrite `config.json` when they run, pointing it at the
  practice or production directories respectively.
- The Flask session key defaults to a dev-only placeholder; set the `FLASK_SECRET_KEY`
  environment variable before serving the API anywhere real.
- Large generated artifacts (`.sav` model files, PNGs, merged CSVs) are committed in this
  repo so the pipeline outputs can be inspected without rerunning everything.
