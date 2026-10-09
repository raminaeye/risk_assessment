"""Reporting: confusion matrix of the deployed model on the test data.

Saves a labeled confusion matrix plot to the model directory.
"""
import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from data_prep import prep_data


###############Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


##############Function for reporting
def score_model(trainmodel_filename='trainedmodel', test_filename='testdata', ext=''):
    """Plot and save the confusion matrix for a trained model.

    Args:
        trainmodel_filename: Base name of the saved model (without
            extension) in the model directory.
        test_filename: Base name of the test CSV (without extension) in
            the test data directory.
        ext: Optional suffix for the saved plot filename.

    Writes ``confusionmatrix{ext}.png`` to the model directory.
    """
    test_df = pd.read_csv(os.path.join(test_data_path, test_filename + '.csv'))
    sc_file = os.path.join(model_path, 'scaler.sav')
    scaler = pickle.load(open(sc_file, 'rb'))
    X, y = prep_data(test_df, scaler)

    model_filename = os.path.join(model_path, trainmodel_filename + '.sav')
    model = pickle.load(open(model_filename, 'rb'))

    y_predict = model.predict(X)
    cm = metrics.confusion_matrix(y, y_predict)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(model_path, 'confusionmatrix' + ext + '.png'))


if __name__ == '__main__':
    score_model()
