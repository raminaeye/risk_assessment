"""Model scoring: F1 score of the trained model on the test data.

The score is written to ``latestscore.txt`` in the model directory so the
drift check in ``fullprocess.py`` can compare runs over time.
"""
import pandas as pd
import pickle
import os
from sklearn import metrics
import json
from data_prep import prep_data


#################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


#################Function for model scoring
def score_model(filename='testdata'):
    """Score the trained model on a test dataset and record the F1 score.

    Args:
        filename: Base name of the test CSV (without extension) in the
            configured test data directory.

    Returns:
        The F1 score as a string. Also written to ``latestscore.txt``.
    """
    print("Testing on " + filename)
    test_df = pd.read_csv(os.path.join(test_data_path, filename + '.csv'))

    sc_file = os.path.join(model_path, 'scaler.sav')
    scaler = pickle.load(open(sc_file, 'rb'))
    X, y = prep_data(test_df, scaler=scaler)

    model_file = os.path.join(model_path, 'trainedmodel.sav')
    print("Loading " + model_file)

    LG = pickle.load(open(model_file, 'rb'))
    y_predict = LG.predict(X)

    f1_score = metrics.f1_score(y, y_predict)
    score_filename = os.path.join(model_path, 'latestscore.txt')

    print("Saving score to " + score_filename)
    f = open(score_filename, 'w')
    f.write(str(f1_score))
    f.close()
    print("f1score {}".format(f1_score))
    return str(f1_score)


if __name__ == '__main__':
    score_model()
