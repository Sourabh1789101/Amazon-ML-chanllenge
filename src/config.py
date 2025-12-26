"""
Configuration file for Amazon ML Challenge - Smart Product Pricing
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Directory paths
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')

# Dataset files
TRAIN_FILE = os.path.join(DATASET_DIR, 'train.csv')
TEST_FILE = os.path.join(DATASET_DIR, 'test.csv')
SAMPLE_TEST_FILE = os.path.join(DATASET_DIR, 'sample_test.csv')
SAMPLE_TEST_OUT_FILE = os.path.join(DATASET_DIR, 'sample_test_out.csv')

# Output files
OUTPUT_FILE = os.path.join(OUTPUTS_DIR, 'test_out.csv')
SAMPLE_OUTPUT_FILE = os.path.join(OUTPUTS_DIR, 'sample_test_out.csv')

# Model files
TEXT_VECTORIZER_FILE = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
PRICE_MODEL_FILE = os.path.join(MODELS_DIR, 'price_model.pkl')
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler.pkl')
FEATURE_EXTRACTOR_FILE = os.path.join(MODELS_DIR, 'feature_extractor.pkl')

# Model parameters
MODEL_CONFIG = {
    # TF-IDF parameters
    'tfidf_max_features': 5000,
    'tfidf_ngram_range': (1, 2),
    'tfidf_min_df': 2,
    'tfidf_max_df': 0.95,
    
    # Random Forest parameters
    'rf_n_estimators': 200,
    'rf_max_depth': 20,
    'rf_min_samples_split': 5,
    'rf_min_samples_leaf': 2,
    'rf_n_jobs': -1,
    'rf_random_state': 42,
    
    # XGBoost parameters
    'xgb_n_estimators': 300,
    'xgb_max_depth': 8,
    'xgb_learning_rate': 0.1,
    'xgb_subsample': 0.8,
    'xgb_colsample_bytree': 0.8,
    'xgb_random_state': 42,
    
    # LightGBM parameters
    'lgbm_n_estimators': 300,
    'lgbm_max_depth': 10,
    'lgbm_learning_rate': 0.1,
    'lgbm_num_leaves': 31,
    'lgbm_random_state': 42,
    
    # Training parameters
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
}

# Image processing parameters
IMAGE_CONFIG = {
    'target_size': (224, 224),
    'batch_size': 32,
    'num_workers': 4,
}

# Text processing parameters
TEXT_CONFIG = {
    'max_length': 512,
    'min_word_freq': 2,
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def create_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in [IMAGES_DIR, MODELS_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ All directories created successfully!")

if __name__ == "__main__":
    create_directories()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Outputs Directory: {OUTPUTS_DIR}")
