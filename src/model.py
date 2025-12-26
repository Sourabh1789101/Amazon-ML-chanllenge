"""
Model Module for Amazon ML Challenge
Contains ML models for price prediction with ensemble methods
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Using fallback models.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM not installed. Using fallback models.")

from config import MODEL_CONFIG, PRICE_MODEL_FILE


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|)/2)
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    float
        SMAPE score (0-200, lower is better)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Avoid division by zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape


class PricePredictor:
    """
    Ensemble model for price prediction using multiple regressors.
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the price predictor.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use: 'rf', 'xgb', 'lgbm', 'ridge', 'ensemble'
        """
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None
        
    def _create_random_forest(self) -> RandomForestRegressor:
        """Create Random Forest regressor."""
        return RandomForestRegressor(
            n_estimators=MODEL_CONFIG['rf_n_estimators'],
            max_depth=MODEL_CONFIG['rf_max_depth'],
            min_samples_split=MODEL_CONFIG['rf_min_samples_split'],
            min_samples_leaf=MODEL_CONFIG['rf_min_samples_leaf'],
            n_jobs=MODEL_CONFIG['rf_n_jobs'],
            random_state=MODEL_CONFIG['rf_random_state']
        )
    
    def _create_xgboost(self):
        """Create XGBoost regressor."""
        if not HAS_XGBOOST:
            print("⚠️ XGBoost not available, using Gradient Boosting instead")
            return GradientBoostingRegressor(
                n_estimators=MODEL_CONFIG['xgb_n_estimators'],
                max_depth=MODEL_CONFIG['xgb_max_depth'],
                learning_rate=MODEL_CONFIG['xgb_learning_rate'],
                random_state=MODEL_CONFIG['xgb_random_state']
            )
        
        return xgb.XGBRegressor(
            n_estimators=MODEL_CONFIG['xgb_n_estimators'],
            max_depth=MODEL_CONFIG['xgb_max_depth'],
            learning_rate=MODEL_CONFIG['xgb_learning_rate'],
            subsample=MODEL_CONFIG['xgb_subsample'],
            colsample_bytree=MODEL_CONFIG['xgb_colsample_bytree'],
            random_state=MODEL_CONFIG['xgb_random_state'],
            n_jobs=-1
        )
    
    def _create_lightgbm(self):
        """Create LightGBM regressor."""
        if not HAS_LIGHTGBM:
            print("⚠️ LightGBM not available, using Gradient Boosting instead")
            return GradientBoostingRegressor(
                n_estimators=MODEL_CONFIG['lgbm_n_estimators'],
                max_depth=MODEL_CONFIG['lgbm_max_depth'],
                learning_rate=MODEL_CONFIG['lgbm_learning_rate'],
                random_state=MODEL_CONFIG['lgbm_random_state']
            )
        
        return lgb.LGBMRegressor(
            n_estimators=MODEL_CONFIG['lgbm_n_estimators'],
            max_depth=MODEL_CONFIG['lgbm_max_depth'],
            learning_rate=MODEL_CONFIG['lgbm_learning_rate'],
            num_leaves=MODEL_CONFIG['lgbm_num_leaves'],
            random_state=MODEL_CONFIG['lgbm_random_state'],
            n_jobs=-1,
            verbose=-1
        )
    
    def _create_ridge(self) -> Ridge:
        """Create Ridge regressor."""
        return Ridge(alpha=1.0, random_state=MODEL_CONFIG['random_state'])
    
    def _create_ensemble(self) -> VotingRegressor:
        """Create ensemble of multiple models."""
        estimators = [
            ('rf', self._create_random_forest()),
            ('ridge', self._create_ridge()),
        ]
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            estimators.append(('xgb', self._create_xgboost()))
        else:
            estimators.append(('gb', GradientBoostingRegressor(
                n_estimators=100, random_state=MODEL_CONFIG['random_state']
            )))
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            estimators.append(('lgbm', self._create_lightgbm()))
        
        return VotingRegressor(estimators=estimators, n_jobs=-1)
    
    def _get_model(self):
        """Get the appropriate model based on model_type."""
        model_factory = {
            'rf': self._create_random_forest,
            'xgb': self._create_xgboost,
            'lgbm': self._create_lightgbm,
            'ridge': self._create_ridge,
            'ensemble': self._create_ensemble,
        }
        
        if self.model_type not in model_factory:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Available: {list(model_factory.keys())}")
        
        return model_factory[self.model_type]()
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Fit the model on training data.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target prices
        verbose : bool
            Print progress information
        """
        if verbose:
            print("=" * 50)
            print(f"🚀 Training {self.model_type.upper()} model")
            print(f"   Training samples: {X.shape[0]}")
            print(f"   Features: {X.shape[1]}")
            print("=" * 50)
        
        # Ensure y is positive (prices should be positive)
        y = np.maximum(y, 0.01)
        
        # Create and fit model
        self.model = self._get_model()
        self.model.fit(X, y)
        
        # Store feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        self.is_fitted = True
        
        if verbose:
            print("✅ Model training complete!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict prices for given features.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Predicted prices
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted! Call fit() first.")
        
        predictions = self.model.predict(X)
        
        # Ensure predictions are positive
        predictions = np.maximum(predictions, 0.01)
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True prices
        verbose : bool
            Print metrics
            
        Returns:
        --------
        Dict with evaluation metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'smape': calculate_smape(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
        }
        
        if verbose:
            print("\n📊 Model Evaluation Metrics:")
            print("-" * 40)
            print(f"   SMAPE: {metrics['smape']:.4f}%")
            print(f"   MAE:   ${metrics['mae']:.2f}")
            print(f"   RMSE:  ${metrics['rmse']:.2f}")
            print(f"   R²:    {metrics['r2']:.4f}")
            print("-" * 40)
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5, verbose: bool = True) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True prices
        cv : int
            Number of folds
        verbose : bool
            Print progress
            
        Returns:
        --------
        Dict with CV scores
        """
        if verbose:
            print(f"\n🔄 Running {cv}-fold Cross-Validation...")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=MODEL_CONFIG['random_state'])
        
        smape_scores = []
        mae_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fresh model for each fold
            fold_model = self._get_model()
            fold_model.fit(X_train, y_train)
            
            predictions = fold_model.predict(X_val)
            predictions = np.maximum(predictions, 0.01)
            
            smape = calculate_smape(y_val, predictions)
            mae = mean_absolute_error(y_val, predictions)
            
            smape_scores.append(smape)
            mae_scores.append(mae)
            
            if verbose:
                print(f"   Fold {fold+1}: SMAPE={smape:.4f}%, MAE=${mae:.2f}")
        
        results = {
            'mean_smape': np.mean(smape_scores),
            'std_smape': np.std(smape_scores),
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
        }
        
        if verbose:
            print(f"\n📈 CV Results:")
            print(f"   SMAPE: {results['mean_smape']:.4f}% (±{results['std_smape']:.4f})")
            print(f"   MAE:   ${results['mean_mae']:.2f} (±${results['std_mae']:.2f})")
        
        return results
    
    def save(self, filepath: str = PRICE_MODEL_FILE):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model!")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'is_fitted': self.is_fitted,
                'feature_importance': self.feature_importance_
            }, f)
        
        print(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = PRICE_MODEL_FILE) -> 'PricePredictor':
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls(model_type=data['model_type'])
        predictor.model = data['model']
        predictor.is_fitted = data['is_fitted']
        predictor.feature_importance_ = data['feature_importance']
        
        print(f"✅ Model loaded from {filepath}")
        return predictor


class ModelSelector:
    """Compare multiple models and select the best one."""
    
    def __init__(self):
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def compare_models(self, X: np.ndarray, y: np.ndarray, 
                       models: List[str] = None, cv: int = 3) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target prices
        models : List[str]
            List of model types to compare
        cv : int
            Number of CV folds
            
        Returns:
        --------
        pd.DataFrame with comparison results
        """
        if models is None:
            models = ['rf', 'ridge', 'xgb', 'lgbm', 'ensemble']
        
        print("=" * 60)
        print("🔍 Model Comparison")
        print("=" * 60)
        
        for model_name in models:
            print(f"\n📌 Testing {model_name.upper()}...")
            try:
                predictor = PricePredictor(model_type=model_name)
                cv_results = predictor.cross_validate(X, y, cv=cv, verbose=False)
                self.results[model_name] = cv_results
                print(f"   SMAPE: {cv_results['mean_smape']:.4f}% (±{cv_results['std_smape']:.4f})")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                self.results[model_name] = {'mean_smape': float('inf')}
        
        # Find best model
        best_name = min(self.results.keys(), key=lambda k: self.results[k]['mean_smape'])
        self.best_model_name = best_name
        
        print("\n" + "=" * 60)
        print(f"🏆 Best Model: {best_name.upper()} (SMAPE: {self.results[best_name]['mean_smape']:.4f}%)")
        print("=" * 60)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results).T
        return results_df
    
    def train_best_model(self, X: np.ndarray, y: np.ndarray) -> PricePredictor:
        """Train the best model on full data."""
        if self.best_model_name is None:
            raise ValueError("Run compare_models() first!")
        
        print(f"\n🚀 Training best model ({self.best_model_name.upper()}) on full data...")
        self.best_model = PricePredictor(model_type=self.best_model_name)
        self.best_model.fit(X, y)
        
        return self.best_model


if __name__ == "__main__":
    # Test with synthetic data
    print("\n🧪 Testing Model Module")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    X = np.random.randn(n_samples, n_features)
    y = np.abs(np.random.randn(n_samples) * 50 + 100)  # Prices around $100
    
    # Test single model
    predictor = PricePredictor(model_type='rf')
    predictor.fit(X[:800], y[:800])
    metrics = predictor.evaluate(X[800:], y[800:])
    
    print("\n📊 Test complete!")
