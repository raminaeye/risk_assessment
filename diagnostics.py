"""Model and data diagnostics: predictions, summary stats, timings, dependencies."""
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import copy
from data_prep import prep_data


##################Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


##################Function to get model predictions
def model_predictions():
    """Predict with the deployed model on the test dataset.

    Returns:
        A list containing the predicted class for each test row.
    """
    LG = pickle.load(open(os.path.join(model_path, 'trainedmodel.sav'), 'rb'))
    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    sc_file = os.path.join(model_path, 'scaler.sav')
    scaler = pickle.load(open(sc_file, 'rb'))
    X, y = prep_data(df, scaler)
    y_predict = list(LG.predict(X))
    return y_predict


##################Function to get summary statistics
def dataframe_summary():
    """Compute summary statistics for the latest ingested dataset.

    For numeric columns: mean, median, standard deviation, and the share
    of non-null values. For text columns: the share of null values.

    Returns:
        Dict mapping each column name to its statistics.
    """
    fversion = open(output_folder_path + '/train_data_versions.txt', 'r').readlines()[-1].replace('\n', '')
    print("Reading " + fversion)
    df = pd.read_csv(fversion)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    list_statistics = {}
    for col in df.columns:
        nan_percent = round(len(df[col].dropna()) / len(df[col]), 3)
        values = list(df[col].dropna())

        if type(values[0]) != str:
            mu = round(np.mean(values), 3)
            med = round(np.median(values), 3)
            std = round(np.std(values), 3)
            list_statistics[col] = [mu, med, std, nan_percent]
        else:
            # Counting NAN values
            nan_percent = 1 - (len(df[col].dropna()) / len(values))
            list_statistics[col] = [nan_percent]
    return list_statistics


##################Function to get timings
def execution_time():
    """Time one run of ingestion and one run of training.

    Returns:
        A string of the two timings in seconds: [ingestion, training].
    """
    training_time = timeit.timeit("train_model()", "from training import train_model", number=1)
    ingestion_time = timeit.timeit("merge_multiple_dataframe()", "from ingestion import merge_multiple_dataframe", number=1)
    times = [(ingestion_time), (training_time)]
    return str(times)


##################Function to check dependencies
def outdated_packages_list():
    """Compare pinned requirements against installed and latest versions.

    Reads ``requirements.txt`` for the pinned versions, then refreshes
    ``requirements_installed.txt`` with the output of
    ``pip list --outdated`` (that file is regenerated on every call, so it
    is an output, not a source of truth). Writes the full comparison to
    ``req_versions.csv``.

    Returns:
        Dataframe with library, current (pinned), latest, and used
        (installed) versions for packages where all four are known.
    """
    list_packages = []
    package_df = {'library': '', 'current': None, 'latest': None, 'used': None}
    texts = open("requirements.txt", 'r').readlines()
    for txt in texts:
        txt = txt.strip()
        if not txt or txt.startswith('#'):
            continue
        library, version = txt.split('==')
        package_df['library'] = library
        package_df['current'] = version
        list_packages.append(copy.copy(package_df))

    df = pd.DataFrame(list_packages)
    # Refresh the outdated-packages snapshot; this file is an output of this
    # function, not an input. Do not edit it by hand.
    subprocess.run(" pip list --outdated > requirements_installed.txt", shell=True)
    texts = open("requirements_installed.txt", 'r').readlines()
    for txt in texts:
        list_of_items = txt.split()
        try:
            library, version, latest, type = list_of_items[0], list_of_items[1], list_of_items[2], list_of_items[3]
            df.loc[df['library'] == library, 'latest'] = latest
            df.loc[df['library'] == library, 'used'] = version
        except (ValueError, IndexError):
            # Header, separator, and footer lines of the pip output.
            pass
    df.dropna().to_csv('req_versions.csv')
    return df.dropna()


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
