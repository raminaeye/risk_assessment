"""Shared data preparation helpers for the risk assessment pipeline.

Builds the feature matrix and target vector used by training, scoring,
reporting, diagnostics, and the API prediction endpoint.
"""
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import json


##################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


##################Feature preparation
def prep_data(df, scaler=None):
    """Extract and scale model features from a dataframe.

    Features are the three numeric columns ``lastmonth_activity``,
    ``lastyear_activity``, and ``number_of_employees``.

    Args:
        df: Dataframe with the feature columns (and ``exited`` when the
            target is needed).
        scaler: A fitted ``StandardScaler``. When None, a new scaler is fit
            on ``df``, saved to the model directory as ``scaler.sav``, and
            used for the transform.

    Returns:
        Tuple ``(X, y)`` where ``X`` is the scaled feature matrix and ``y``
        is the target array, or None when the dataframe has no ``exited``
        column (e.g. pure prediction input).
    """
    LM_A = list(df['lastmonth_activity'])
    LY_A = list(df['lastyear_activity'])
    N_E = list(df['number_of_employees'])

    X = np.asarray([LM_A, LY_A, N_E]).T

    if scaler is None:
        # First-time fit: learn the scaling parameters and persist them so
        # every later use (scoring, API predictions) scales identically.
        scaler = StandardScaler()
        scaler.fit(X)
        sc_file = os.path.join(model_path, 'scaler.sav')
        pickle.dump(scaler, open(sc_file, 'wb'))

    # Transform exactly once, whether the scaler was just fit or passed in.
    X = scaler.transform(X)

    y = np.asarray(list(df['exited'])) if 'exited' in df.columns else None
    return X, y
