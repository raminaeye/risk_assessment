"""Flask API for the risk assessment model.

Endpoints:
    /prediction?filename=<csv>  Predict churn on a CSV of company records.
    /scoring                    F1 score of the deployed model on test data.
    /summarystats               Summary statistics of the training data.
    /diagnostics                 Summary stats, timings, and outdated-package info.
"""
from flask import Flask, request
import pandas as pd
import pickle
import json
import os
from scoring import score_model
from diagnostics import dataframe_summary, execution_time, outdated_packages_list
from data_prep import prep_data


######################Set up variables for use in our script
app = Flask(__name__)
# Session signing key. Set FLASK_SECRET_KEY in the environment for any real
# deployment; the fallback is a dev-only placeholder and must not be used
# in production.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-only-secret-key')

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])

prediction_model = pickle.load(open(os.path.join(model_path, 'trainedmodel.sav'), 'rb'))
scaler = pickle.load(open(os.path.join(model_path, 'scaler.sav'), 'rb'))


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET', 'OPTIONS'])
def prediction():
    """Predict churn for each row of the requested CSV file.

    Query param:
        filename: CSV in the ingested data directory.

    Returns the model's predicted labels as a string. Input features are
    scaled with the same scaler used in training.
    """
    filename = request.args.get('filename')
    df = pd.read_csv(os.path.join(dataset_csv_path, filename))
    X, _ = prep_data(df, scaler)
    predictions = prediction_model.predict(X)
    return str(predictions)


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    """Return the F1 score of the deployed model on the test data."""
    return score_model()


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def summarystats():
    """Return means, medians, standard deviations, and null shares per column."""
    return dataframe_summary()


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    """Return summary stats, ingestion/training timings, and outdated packages."""
    summary_stats = dataframe_summary()
    timing = execution_time()
    outdated = outdated_packages_list().values
    result = [str(summary_stats), str(timing), str(outdated)]
    return str([result[0], '\n', result[1], '\n', result[2]])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
