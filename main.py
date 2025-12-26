"""
Main entry point for Amazon ML Challenge
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Amazon ML Challenge - Smart Product Pricing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --quick              Quick training on sample data
  python main.py train --model ensemble     Train ensemble model
  python main.py train --compare            Compare all models
  python main.py predict --sample           Predict on sample test file
  python main.py predict                    Predict on full test file
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train price prediction model')
    train_parser.add_argument('--train-path', type=str, default='dataset/train.csv',
                             help='Path to training CSV file')
    train_parser.add_argument('--model', type=str, default='ensemble',
                             choices=['rf', 'xgb', 'lgbm', 'ridge', 'ensemble'],
                             help='Model type to train')
    train_parser.add_argument('--compare', action='store_true',
                             help='Compare multiple models')
    train_parser.add_argument('--cv-folds', type=int, default=5,
                             help='Number of cross-validation folds')
    train_parser.add_argument('--quick', action='store_true',
                             help='Quick training on sample data')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--test-path', type=str, default='dataset/test.csv',
                               help='Path to test CSV file')
    predict_parser.add_argument('--output-path', type=str, default='outputs/test_out.csv',
                               help='Path to save predictions')
    predict_parser.add_argument('--sample', action='store_true',
                               help='Use sample test file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from src.train import train_pipeline, quick_train_sample
        
        if args.quick:
            quick_train_sample()
        else:
            train_pipeline(
                train_path=args.train_path,
                model_type=args.model,
                compare_models=args.compare,
                cv_folds=args.cv_folds
            )
    
    elif args.command == 'predict':
        from src.predict import predict_pipeline, predict_sample
        
        if args.sample:
            predict_sample()
        else:
            predict_pipeline(
                test_path=args.test_path,
                output_path=args.output_path
            )
    
    else:
        parser.print_help()
        print("\n🚀 Quick Start:")
        print("   1. Train: python main.py train --quick")
        print("   2. Predict: python main.py predict --sample")


if __name__ == "__main__":
    main()
