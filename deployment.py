"""Model deployment: promote the trained model and its artifacts.

Copies the latest model pickle, the scaler, the score record, and the
ingested-files record into the deployment directory.
"""
import pickle
import os
import json
import subprocess


##################Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def store_model_into_pickle():
    """Copy the trained model, scaler, score, and ingest record to the deployment directory."""
    mdl_pkl = os.path.join(model_path, 'trainedmodel.sav')
    sc_file = os.path.join(model_path, 'scaler.sav')
    scaler = pickle.load(open(sc_file, 'rb'))
    model = pickle.load(open(mdl_pkl, 'rb'))
    ing_txt = os.path.join(output_folder_path, 'ingestedfiles.txt')
    lsc_txt = os.path.join(model_path, 'latestscore.txt')
    print("Deploying from model trained from data from {} with the score file saved in {}".format(ing_txt, lsc_txt))
    subprocess.call(['cp', ing_txt, prod_deployment_path + '/'])
    subprocess.call(['cp', lsc_txt, prod_deployment_path + '/'])
    pickle.dump(model, open(os.path.join(prod_deployment_path, 'trainedmodel.sav'), 'wb'))
    pickle.dump(scaler, open(os.path.join(prod_deployment_path, 'scaler.sav'), 'wb'))


if __name__ == '__main__':
    store_model_into_pickle()
