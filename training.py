"""Model training: fit a logistic regression on the ingested training data."""
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
from data_prep import prep_data


###################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for training the model
def train_model(ext=''):
    """Train the churn model and save it to the model directory.

    Reads the latest training file listed in ``train_data_versions.txt``,
    prepares features with ``prep_data`` (fitting and saving a new scaler),
    fits a logistic regression, and writes the pickled model to
    ``trainedmodel{ext}.sav``.

    Args:
        ext: Optional suffix for the saved model filename, used when
            training a challenger model during the drift check.
    """
    fversion = open(output_folder_path + '/train_data_versions.txt', 'r').readlines()[-1].replace('\n', '')
    print("Reading " + fversion)
    df = pd.read_csv(fversion)
    X, y = prep_data(df)

    print("Training input shape {} and output shape {}".format(X.shape, y.shape))
    # use this logistic regression for training
    LG = LogisticRegression(C=1.0, solver='liblinear', max_iter=500).fit(X, y)
    # write the trained model to the workspace
    model_file = os.path.join(model_path, 'trainedmodel' + ext + '.sav')
    print("Saving " + model_file)
    pickle.dump(LG, open(model_file, 'wb'))


if __name__ == '__main__':
    train_model()
