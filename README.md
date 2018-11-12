# eegKeras
EEG Analysis scripts with Keras on TensorFlow

# How to use these tests

## Setup

The top level scripts are in the mainScripts/ directory.  There are 4 top level scripts:
1) train.py
2) test.py
3) extractFeatures_CHB.py
4) extractFeatures_TUH.py
5) applyPCAtoCSVfile.py

Before running any of the top level script, add the path to this directory to the PYTHONPATH environment variable.

On Mac OSX issue the following command in the terminal:

export PYTHONPATH=$PYTHONPATH:\<path-to-the-top-directory\>

On Windows issue the following command:

TBD

## Dataset directories

The eegAnalysis tools were created to analyze 2 datasets:
1) CHB-MIT dataset
2) TUH dataset

Both these datasets are large in size (tens of GB) and have their own conventions on the organizatoin of the patient specific EEG records.  The only common aspect between these two datasets is that all the EEG files are EDF (European Data Format) format.

For the analysis tools in this repository, the following directory conventions are used:

* ### Top level directory for all the data and out files:
    */Users/guest/Documents/Data*

* ### Top level directory for CHB-MIT EDF data files:
    */Users/guest/Documents/Data/chb_eeg_data/*

* ### Top level directory for TUH EDF data files:
    */Users/guest/Documents/Data/tuh_eeg_data/*

* ### Top level directory for CHB-MIT output files:
    */Users/guest/Documents/Data/chb_eeg_output/*

* ### Top level directory for TUH output files:
    */Users/guest/Documents/Data/tuh_eeg_output/*

* ### Top level diretory for saved model files:
    */Users/guest/Documents/Data/savedModels/*

## Step 1: Extract LineLength and FFT features from both the CHB-MIT and TUH datasets

## Step 2: Verify that the output files from step 1 are of the Pandas Dataframe format, not Numpy ndarray format.

Pandas dataframe format csv files have a header row with the column names and an index column (left most column) with the row numbers.

If the LineLength and/or FFT are of numpy ndarray format, use the script extractFeatures_*.py with the command "convertToPandas".

## Step 3: Create LSTM model and weights file(s) after training the feature file(s)

For Example:
Use the command python "./train.py ./train_mac.json LSTM_CHB_LL"  to create the model files for LSTM on CHB dataset's LineLength feature files.

## Step 4: Use the above LSTM model file(s) to create intermediate LSTM predicted csv files.

## Step 5: Create DNN model file(s) by training the above created LSTM-predicted datasets.

## Step 6:  Run the hybrid test using the LSTM and DNN models created in steps 3 and 5 above.



