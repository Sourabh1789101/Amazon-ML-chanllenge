"""
Training Pipeline for Amazon ML Challenge
End-to-end training workflow for price prediction
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAIN_FILE, TEST_FILE, SAMPLE_TEST_FILE,
    MODELS_DIR, OUTPUTS_DIR, MODEL_CONFIG,
    create_directories
)
from data_preprocessing import preprocess_dataframe, load_and_preprocess_data
from feature_extraction import FeatureEngineer
from model import PricePredictor, ModelSelector, calculate_smape


def train_pipeline(
    train_path: str = TRAIN_FILE,
    model_type: str = 'ensemble',
    compare_models: bool = False,
    cv_folds: int = 5,
    verbose: bool = True
) -> PricePredictor:
    """
    Complete training pipeline.
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV
    model_type : str
        Type of model ('rf', 'xgb', 'lgbm', 'ridge', 'ensemble')
    compare_models : bool
        Whether to compare multiple models first
    cv_folds : int
        Number of cross-validation folds
    verbose : bool
        Print progress information
        
    Returns:
    --------
    PricePredictor
        Trained model
    """
    print("=" * 70)
    print("🚀 AMAZON ML CHALLENGE - TRAINING PIPELINE")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # Step 1: Load and preprocess data
    print("\n📚 STEP 1: Loading and Preprocessing Data")
    print("-" * 50)
    
    train_df = pd.read_csv(train_path)
    print(f"✅ Loaded {len(train_df)} training samples")
    
    # Check for price column
    if 'price' not in train_df.columns:
        raise ValueError("Training data must contain 'price' column!")
    
    # Preprocess
    train_df = preprocess_dataframe(train_df)
    
    # Get target variable
    y = train_df['price'].values
    print(f"📊 Price statistics:")
    print(f"   Min: ${y.min():.2f}")
    print(f"   Max: ${y.max():.2f}")
    print(f"   Mean: ${y.mean():.2f}")
    print(f"   Median: ${np.median(y):.2f}")
    
    # Step 2: Feature Engineering
    print("\n🔧 STEP 2: Feature Engineering")
    print("-" * 50)
    
    feature_engineer = FeatureEngineer()
    X = feature_engineer.fit_transform(train_df)
    
    print(f"✅ Feature matrix created: {X.shape}")
    
    # Step 3: Model Selection (optional)
    if compare_models:
        print("\n🔍 STEP 3: Model Comparison")
        print("-" * 50)
        
        selector = ModelSelector()
        comparison_df = selector.compare_models(X, y, cv=min(cv_folds, 3))
        print("\n📊 Model Comparison Results:")
        print(comparison_df)
        
        # Train best model
        model = selector.train_best_model(X, y)
    else:
        # Step 3: Train specified model
        print(f"\n🎯 STEP 3: Training {model_type.upper()} Model")
        print("-" * 50)
        
        model = PricePredictor(model_type=model_type)
        
        # Cross-validation
        if cv_folds > 1:
            cv_results = model.cross_validate(X, y, cv=cv_folds, verbose=verbose)
        
        # Train on full data
        model.fit(X, y, verbose=verbose)
    
    # Step 4: Save model and feature extractors
    print("\n💾 STEP 4: Saving Models")
    print("-" * 50)
    
    feature_engineer.save()
    model.save()
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return model


def quick_train_sample(verbose: bool = True):
    """
    Quick training on sample data for testing.
    """
    print("\n🧪 Quick Training on Sample Data")
    print("=" * 50)
    
    # Use sample test file for quick testing
    sample_df = pd.read_csv(SAMPLE_TEST_FILE)
    
    # Create synthetic prices for testing
    np.random.seed(42)
    sample_df['price'] = np.random.uniform(5, 200, len(sample_df))
    
    # Preprocess
    sample_df = preprocess_dataframe(sample_df)
    
    # Feature engineering
    feature_engineer = FeatureEngineer(max_tfidf_features=500)
    X = feature_engineer.fit_transform(sample_df)
    y = sample_df['price'].values
    
    # Quick model training
    model = PricePredictor(model_type='rf')
    model.fit(X[:int(len(X)*0.8)], y[:int(len(y)*0.8)], verbose=verbose)
    
    # Evaluate
    metrics = model.evaluate(X[int(len(X)*0.8):], y[int(len(y)*0.8):])
    
    # Save
    feature_engineer.save()
    model.save()
    
    print("\n✅ Quick training complete!")
    return model


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train Amazon Price Prediction Model')
    
    parser.add_argument('--train-path', type=str, default=TRAIN_FILE,
                       help='Path to training CSV file')
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['rf', 'xgb', 'lgbm', 'ridge', 'ensemble'],
                       help='Model type to train')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models before training')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training on sample data')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_train_sample(verbose=args.verbose)
    else:
        # Check if training file exists
        if not os.path.exists(args.train_path):
            print(f"⚠️ Training file not found: {args.train_path}")
            print("   Running quick training on sample data instead...")
            quick_train_sample(verbose=args.verbose)
        else:
            train_pipeline(
                train_path=args.train_path,
                model_type=args.model,
                compare_models=args.compare,
                cv_folds=args.cv_folds,
                verbose=args.verbose
            )


if __name__ == "__main__":
    main()
