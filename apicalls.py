"""Call every API endpoint and save the combined responses.

Hits /prediction, /scoring, /summarystats, and /diagnostics on the local
Flask app and writes all responses to ``apireturns{ext}.txt`` in the model
directory. The API (app.py) must be running on port 8000 first.
"""
import requests
import json
import os
import ast


# Specify a URL that resolves to your workspace
BASE_URL = "http://127.0.0.1:8000"


def call(datafile='finaldata', ext=''):
    """Call each API endpoint and store the responses.

    Args:
        datafile: Base name of the CSV (without extension) to send to the
            /prediction endpoint.
        ext: Suffix for the output file, ``apireturns{ext}.txt``.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    model_path = os.path.join(config['output_model_path'])

    # Call each API endpoint and store the responses
    response0 = requests.get(BASE_URL + '/prediction?filename=' + datafile + '.csv').content
    response1 = requests.get(BASE_URL + '/scoring').content
    response2 = requests.get(BASE_URL + '/summarystats').content
    response3 = requests.get(BASE_URL + '/diagnostics').content
    # The diagnostics endpoint returns a stringified list of
    # [summary, timing, outdated-packages]; pull the pieces back out.
    response30 = ast.literal_eval(response3.decode())[0]
    response31 = ast.literal_eval(response3.decode())[2]
    response32 = ' '.join(ast.literal_eval(response3.decode())[3:])

    # combine all API responses
    responses = '\n'.join([
        BASE_URL + '/prediction?filename=' + datafile + '.csv', response0.decode(),
        BASE_URL + '/scoring', response1.decode(),
        BASE_URL + '/summarystats', response2.decode(),
        BASE_URL + '/diagnostics', 'summary', response30,
        BASE_URL + '/diagnostics', 'timing', response31,
        BASE_URL + '/diagnostics', 'outdated', response32])

    # write the responses to your workspace
    with open(os.path.join(model_path, 'apireturns' + ext + '.txt'), 'w') as f:
        f.write(responses)


if __name__ == "__main__":
    call(datafile='finaldata2', ext='')
