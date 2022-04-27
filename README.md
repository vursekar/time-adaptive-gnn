# 10-707 Final Project: Time-Adaptive GNNs



## bike_share folder (for final report)

construct_full_dataset.ipynb provides code to compile the raw csv files into time-binned counts

models is a folder containing all used models as classes 

helper_utils is a folder containing utilities for data preprocessing

model-test.ipynb provides code to import and train models



## traffic_forecasting folder (for midway report)

common.py contains constants (dataset file paths and such) to be used throughout this project.

util.py contains utility functions/classes to be used throughout this project.

preprocessing.ipynb loads dataset and performs data preprocessing.

model_X.ipynb for any X (e.g. Feedforward, LinearRegression) trains a model and generates predictions, then saves/serializes them.

evaluate.ipynb loads dataset, loads model, and evaluates model (specified by model_name within) on dataset.

evaluate_multiple.ipynb is like evaluate.ipynb, but it can load and evaluate multiple models at once.
