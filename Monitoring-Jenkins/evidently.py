## Script non-alert
import os
import argparse
import pandas as pd
#from sklearn import datasets
#from evidently.report import Report
#import evidently
import pathlib
import logging
import boto3
import shutil
#from evidently.metric_preset import DataDriftPreset
#from evidently.test_preset import DataDriftTestPreset
import pip
# Define the library name
#library_name = 'evidently'
# Install the library using pip
#pip.main(['install', library_name])
import evidently
library_name = 'evidently'
submodule_names = ['test_suite']

# Install the library and submodules using pip
pip.main(['install', f'{library_name}[{",".join(submodule_names)}]'])
#from evidently.tests import *

#from evidently import TestSuite
#from evidently import ColumnMapping
#from evidently.test_suite import TestSuite
#from evidently.test_preset import DataStabilityTestPreset
#from evidently.test_preset import DataQualityTestPreset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Data Drift Script')
    parser.add_argument('--baseline', type=str, help='S3 path to the baseline dataset')
    parser.add_argument('--test', type=str, help='S3 path to the test dataset')

    # Parse the command-line arguments
    args = parser.parse_args()
    current_dir = os.getcwd()
    src_file = f'{current_dir}/functions.py'
    dest_dir = '/opt/ml/processing/input/code/'
    shutil.copy(src_file, dest_dir)

    from functions import import_libs
    #import_libs()
    #TestSuite = import_
    
    
    
    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    baseline = args.baseline
    pathlib.Path(f"{base_dir}/data1").mkdir(parents=True, exist_ok=True)

    test = args.test
    
    bucket = baseline.split("/")[2]
    key_baseline = "/".join(baseline.split("/")[3:])
    key_test = "/".join(test.split("/")[3:])
    

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key_baseline)
    logger.info("Downloading data from bucket: %s, key: %s", bucket, key_test)
    
    fn_baseline = f"{base_dir}/data/baseline.csv"
    fn_test = f"{base_dir}/data1/test.csv"
    
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key_baseline, fn_baseline)
    s3.Bucket(bucket).download_file(key_test, fn_test)
    

    logger.debug("Reading downloaded data.")
    df_train = pd.read_csv(
        fn_baseline
    )
    df_test = pd.read_csv(
        fn_test,names=df_train.columns
    )    

    # Read the CSV files from S3
    #df_train = pd.read_csv(args.baseline)
    #df_test = pd.read_csv(args.test, names=df_train.columns)



    #df_train = pd.read_csv("s3://sm-dashboard-t/sagemaker-modelmonitor/train_headers/train_data_with_headers.csv")
    #df_test = pd.read_csv("s3://sm-dashboard-t/sagemaker-modelmonitor/test_data/test_data.csv",names=df_train.columns)
    #df_train = df_train[:300]
    #df_train_2 = df_train[-300:]
    data_drift = TestSuite(
    tests=[
        DataDriftTestPreset(stattest="psi"),
    ]
    )

    data_drift.run(reference_data=df_train, current_data=df_test)

    test_summary = data_drift.as_dict()
    other_tests_summary = []

    failed_tests = []
    for test in test_summary["tests"]:
        if test["status"].lower() == "fail":
            failed_tests.append(test)

    is_alert = any([failed_tests, other_tests_summary])
    alert_status = f"Alert Detected: {is_alert}"
    print(f"Alert Detected: {is_alert}")