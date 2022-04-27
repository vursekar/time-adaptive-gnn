import os

# CONSTANTS (modify as needed)

DATA_DIRECTORY = "/Users/varunursekar/Projects/advDeepLearning/final_project/raw_data"
#DATA_DIRECTORY = r"C:\Users\shali\Documents\CMU_10-707_Project\data"
# DATASET_NAME = "PEMS-BAY"
DATASET_NAME = "METR-LA"
OUTPUT_DIRECTORY = "/Users/varunursekar/Projects/advDeepLearning/final_project/processed_data"
PLOTS_DIRECTORY = "/Users/varunursekar/Projects/advDeepLearning/final_project/plots"
# SENSOR_LOCATIONS_FILE = '/Users/varunursekar/Projects/advDeepLearning/final_project/DCRNN-master/data/sensor_graph/graph_sensor_locations_bay.csv'
SENSOR_LOCATIONS_FILE = '/Users/varunursekar/Projects/advDeepLearning/final_project/DCRNN-master/data/sensor_graph/graph_sensor_locations.csv'

PREPROCESSED_DATASET_FILENAME = DATASET_NAME + "_preprocessed.npz"
PREPROCESSING_SCALER_FILENAME = DATASET_NAME + "_scaler.pk"

DATASET_DIRECTORY = os.path.join(DATA_DIRECTORY, DATASET_NAME)
PREPROCESSED_DATASET_FILEPATH = os.path.join(OUTPUT_DIRECTORY, PREPROCESSED_DATASET_FILENAME)
PREPROCESSING_SCALER_FILEPATH = os.path.join(OUTPUT_DIRECTORY, PREPROCESSING_SCALER_FILENAME)

def model_name_to_model_filepath(model_name):
    filename = "_".join([DATASET_NAME, model_name, "model"]) + ".h5"
    return os.path.join(OUTPUT_DIRECTORY, filename)

def model_name_to_run_info_filepath(model_name):
    filename = "_".join([DATASET_NAME, model_name, "run_info"]) + ".pk"
    return os.path.join(OUTPUT_DIRECTORY, filename)
