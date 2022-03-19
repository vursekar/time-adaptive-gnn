# 10-707 Final Project: Traffic Forecasting

common.py contains constants (dataset file paths and such) to be used throughout this project.

util.py contains utility functions/classes to be used throughout this project.

preprocessing.ipynb loads dataset and performs data preprocessing.

model_X.ipynb for any X (e.g. Feedforward, LinearRegression) trains a model and generates predictions, then saves/serializes them.

evaluate.ipynb loads dataset, loads model, and evaluates model (specified by model_name within) on dataset.

evaluate_multiple.ipynb is like evaluate.ipynb, but it can load and evaluate multiple models at once.
