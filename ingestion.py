"""Data ingestion: merge source CSVs into a single training-ready dataset.

Keeps a record of which files have been ingested so repeated runs only
pick up new data.
"""
import pandas as pd
import os
import json


#############Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
prod_deployment_path = config['prod_deployment_path']
test_data_path = config['test_data_path']


#############Track which source files have been ingested
def ingestion(WRITE_APPEND='w'):
    """Record newly seen source files in ``ingestedfiles.txt``.

    Args:
        WRITE_APPEND: File mode for the record file. ``'w'`` resets the
            record (first run); ``'a'`` appends only files not seen before.
    """
    files_input_folder = [input_folder_path + '/' + f
                          for f in os.listdir(input_folder_path) if 'csv' in f]
    try:
        files_ingested = [f.replace('\n', '') for f in
                          open(output_folder_path + '/ingestedfiles.txt', 'r').readlines()]
    except (FileNotFoundError, OSError):
        # No record yet: everything in the source folder is new.
        files_ingested = []
    ingestF = open(output_folder_path + '/ingestedfiles.txt', WRITE_APPEND)
    for f in files_input_folder:
        if f not in files_ingested:
            ingestF.write(f)
            ingestF.write('\n')
    ingestF.close()


#############Function for data ingestion
def merge_multiple_dataframe(filename_input='ingestedfiles', filename_output='finaldata', testing=0):
    """Concatenate the recorded CSVs into one deduplicated dataframe.

    Args:
        filename_input: Base name of the text file listing CSVs to merge.
        filename_output: Base name for the merged output CSV.
        testing: When 0, writes the training dataset and records it in
            ``train_data_versions.txt``. When 1, writes a test dataset to
            the test data directory and records it in
            ``test_data_versions.txt``.

    Returns:
        The merged, deduplicated dataframe.
    """
    if not testing:
        ftrainversion = open(output_folder_path + '/train_data_versions.txt', 'w')
    else:
        ftestversion = open(output_folder_path + '/test_data_versions.txt', 'w')

    ingestF = open(output_folder_path + '/' + filename_input + '.txt', 'r')
    all_files = ingestF.readlines()
    # check for datasets, compile them together, and write to an output file
    list_of_dfs = []
    for f in all_files:
        print(f)
        f = f.replace('\n', '')
        df = pd.read_csv(f)
        list_of_dfs.append(df)
    df_merged = pd.concat(list_of_dfs).reset_index().drop('index', axis=1)
    df_merged.drop_duplicates(inplace=True)

    if not testing:
        latest_datafile = output_folder_path + '/' + filename_output + str(len(all_files)) + '.csv'
        print("Saving " + latest_datafile)
        ftrainversion.write(latest_datafile)
        fversion = ftrainversion
    else:
        latest_datafile = test_data_path + '/' + filename_output + '.csv'
        print("Saving " + latest_datafile)
        ftestversion.write(latest_datafile)
        fversion = ftestversion

    fversion.write('\n')
    df_merged.to_csv(latest_datafile)
    ingestF.close()
    fversion.close()
    return df_merged


if __name__ == '__main__':
    ingestion()
    merge_multiple_dataframe()
