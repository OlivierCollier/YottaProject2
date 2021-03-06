import os

# Directories
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TRAIN_DATA_DIR = os.path.join(REPO_DIR, "data/train/")
TEST_DATA_DIR = os.path.join(REPO_DIR, "data/test/")
MODEL_DIR = os.path.join(REPO_DIR, "models/")
TRAINED_MODEL_FILENAME = "trained_model.hdf5"
FINAL_MODEL_FILENAME = "PCR_model.hdf5"

# Initialize random seed
SEED = 1234567890

# Data constants
CLASSES = ["Amsterdam", "London", "Paris", "Strasbourg", "Venice"]
IMAGE_SHAPE = (150, 150, 3)
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
INTERPOLATION = "bilinear"

# Training constants
EPOCHS = 100
EPOCHS_RETRAIN = 10

# Training schedule
INITIAL_LEARNING_RATE = 0.001
FINAL_LEARNING_RATE = 0.0001
DECAY_STEPS = 1000
FINE_TUNING_LEARNING_RATE = 1.3e-5

# Data scrapping (only if you want to scrap new images from google image)
DRIVER_PATH = "path/to/google chrome/drivers"
IMAGES_PATH = "path/to/download/folder"
N_IMAGES = 100
QUERIES = [
    "facade paris",
    "building front paris",
    "maison typique Paris",
    "typical house Paris",
    "logement Paris",
    "housing Paris",
    "batiment Paris",
    "building Paris",
    "house Paris",
    "maison Paris",
]  # examples of queries for Paris
