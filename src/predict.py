"""
Prediction Pipeline for Amazon ML Challenge
Generate predictions for test data
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TEST_FILE, SAMPLE_TEST_FILE, OUTPUT_FILE, SAMPLE_OUTPUT_FILE,
    MODELS_DIR, OUTPUTS_DIR, create_directories
)
from data_preprocessing import preprocess_dataframe
from feature_extraction import FeatureEngineer
from model import PricePredictor


def predict_pipeline(
    test_path: str = TEST_FILE,
    output_path: str = OUTPUT_FILE,
    batch_size: int = 10000,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Complete prediction pipeline.
    
    Parameters:
    -----------
    test_path : str
        Path to test data CSV
    output_path : str
        Path to save predictions
    batch_size : int
        Batch size for processing large datasets
    verbose : bool
        Print progress information
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sample_id and predicted prices
    """
    print("=" * 70)
    print("🎯 AMAZON ML CHALLENGE - PREDICTION PIPELINE")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # Step 1: Load models
    print("\n📂 STEP 1: Loading Trained Models")
    print("-" * 50)
    
    try:
        feature_engineer = FeatureEngineer.load()
        model = PricePredictor.load()
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found. Please run training first!")
        print(f"   Run: python src/train.py")
        raise e
    
    print("✅ Models loaded successfully!")
    
    # Step 2: Load test data
    print("\n📚 STEP 2: Loading Test Data")
    print("-" * 50)
    
    test_df = pd.read_csv(test_path)
    total_samples = len(test_df)
    print(f"✅ Loaded {total_samples} test samples")
    
    # Step 3: Process and predict in batches
    print("\n🔄 STEP 3: Processing and Predicting")
    print("-" * 50)
    
    all_predictions = []
    all_sample_ids = []
    
    # Process in batches for memory efficiency
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        
        batch_df = test_df.iloc[start_idx:end_idx].copy()
        
        # Preprocess batch
        batch_processed = preprocess_dataframe(batch_df)
        
        # Extract features
        X_batch = feature_engineer.transform(batch_processed)
        
        # Predict
        predictions = model.predict(X_batch)
        
        all_predictions.extend(predictions)
        all_sample_ids.extend(batch_df['sample_id'].tolist())
        
        if verbose and batch_idx % 5 == 0:
            print(f"   Processed batch {batch_idx + 1}/{num_batches}")
    
    # Step 4: Create output DataFrame
    print("\n📝 STEP 4: Creating Output File")
    print("-" * 50)
    
    output_df = pd.DataFrame({
        'sample_id': all_sample_ids,
        'price': all_predictions
    })
    
    # Ensure positive prices
    output_df['price'] = output_df['price'].apply(lambda x: max(x, 0.01))
    
    # Validate output
    print(f"✅ Generated {len(output_df)} predictions")
    print(f"📊 Prediction statistics:")
    print(f"   Min:    ${output_df['price'].min():.2f}")
    print(f"   Max:    ${output_df['price'].max():.2f}")
    print(f"   Mean:   ${output_df['price'].mean():.2f}")
    print(f"   Median: ${output_df['price'].median():.2f}")
    
    # Step 5: Save output
    print(f"\n💾 STEP 5: Saving to {output_path}")
    print("-" * 50)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"✅ Predictions saved to {output_path}")
    
    # Verify output format
    print("\n📋 Output file preview:")
    print(output_df.head(10).to_string(index=False))
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ PREDICTION COMPLETE!")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Output file: {output_path}")
    print(f"   Total predictions: {len(output_df)}")
    print("=" * 70)
    
    return output_df


def predict_single(catalog_content: str, image_link: str = None) -> float:
    """
    Predict price for a single product.
    
    Parameters:
    -----------
    catalog_content : str
        Product catalog content
    image_link : str, optional
        URL to product image
        
    Returns:
    --------
    float
        Predicted price
    """
    # Load models
    feature_engineer = FeatureEngineer.load()
    model = PricePredictor.load()
    
    # Create single-row DataFrame
    df = pd.DataFrame({
        'sample_id': [0],
        'catalog_content': [catalog_content],
        'image_link': [image_link if image_link else '']
    })
    
    # Preprocess
    df = preprocess_dataframe(df)
    
    # Extract features
    X = feature_engineer.transform(df)
    
    # Predict
    price = model.predict(X)[0]
    
    return max(price, 0.01)


def predict_sample():
    """Generate predictions for sample test file."""
    print("\n🧪 Generating predictions for sample test file")
    
    return predict_pipeline(
        test_path=SAMPLE_TEST_FILE,
        output_path=SAMPLE_OUTPUT_FILE,
        verbose=True
    )


def main():
    """Main entry point for prediction."""
    parser = argparse.ArgumentParser(description='Generate Price Predictions')
    
    parser.add_argument('--test-path', type=str, default=TEST_FILE,
                       help='Path to test CSV file')
    parser.add_argument('--output-path', type=str, default=OUTPUT_FILE,
                       help='Path to save predictions')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Batch size for processing')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample test file')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.sample:
        predict_sample()
    else:
        # Check if test file exists
        if not os.path.exists(args.test_path):
            print(f"⚠️ Test file not found: {args.test_path}")
            print("   Using sample test file instead...")
            predict_sample()
        else:
            predict_pipeline(
                test_path=args.test_path,
                output_path=args.output_path,
                batch_size=args.batch_size,
                verbose=args.verbose
            )


if __name__ == "__main__":
    main()
