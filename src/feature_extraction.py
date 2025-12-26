"""
Feature Extraction Module for Amazon ML Challenge
Handles text vectorization and feature engineering
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, TEXT_VECTORIZER_FILE, SCALER_FILE


class TextFeatureExtractor:
    """Extract features from text using TF-IDF and custom features."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = None
        self.unit_encoder = None
        self.is_fitted = False
        
    def fit(self, texts: pd.Series, units: pd.Series = None):
        """
        Fit the feature extractors on training data.
        
        Parameters:
        -----------
        texts : pd.Series
            Text data to fit TF-IDF on
        units : pd.Series, optional
            Unit data for label encoding
        """
        print("🔄 Fitting TF-IDF vectorizer...")
        
        # Initialize and fit TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=MODEL_CONFIG['tfidf_min_df'],
            max_df=MODEL_CONFIG['tfidf_max_df'],
            stop_words='english',
            sublinear_tf=True
        )
        
        # Clean texts and fit
        clean_texts = texts.fillna('').astype(str)
        self.tfidf_vectorizer.fit(clean_texts)
        
        # Fit unit encoder if provided
        if units is not None:
            print("🔄 Fitting unit encoder...")
            self.unit_encoder = LabelEncoder()
            self.unit_encoder.fit(units.fillna('unknown').astype(str))
        
        self.is_fitted = True
        print(f"✅ Feature extractor fitted! TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
    def transform(self, texts: pd.Series, units: pd.Series = None) -> np.ndarray:
        """
        Transform texts to TF-IDF features.
        
        Parameters:
        -----------
        texts : pd.Series
            Text data to transform
        units : pd.Series, optional
            Unit data to encode
            
        Returns:
        --------
        np.ndarray
            TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted! Call fit() first.")
        
        clean_texts = texts.fillna('').astype(str)
        tfidf_features = self.tfidf_vectorizer.transform(clean_texts).toarray()
        
        # Add unit encoding if available
        if units is not None and self.unit_encoder is not None:
            unit_clean = units.fillna('unknown').astype(str)
            # Handle unseen labels
            unit_encoded = []
            for u in unit_clean:
                try:
                    unit_encoded.append(self.unit_encoder.transform([u])[0])
                except ValueError:
                    unit_encoded.append(-1)  # Unknown unit
            unit_features = np.array(unit_encoded).reshape(-1, 1)
            tfidf_features = np.hstack([tfidf_features, unit_features])
        
        return tfidf_features
    
    def fit_transform(self, texts: pd.Series, units: pd.Series = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts, units)
        return self.transform(texts, units)
    
    def save(self, filepath: str = TEXT_VECTORIZER_FILE):
        """Save the fitted extractors to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'unit_encoder': self.unit_encoder,
                'is_fitted': self.is_fitted,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range
            }, f)
        print(f"✅ Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = TEXT_VECTORIZER_FILE) -> 'TextFeatureExtractor':
        """Load a fitted extractor from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls(
            max_features=data['max_features'],
            ngram_range=data['ngram_range']
        )
        extractor.tfidf_vectorizer = data['tfidf_vectorizer']
        extractor.unit_encoder = data['unit_encoder']
        extractor.is_fitted = data['is_fitted']
        
        print(f"✅ Feature extractor loaded from {filepath}")
        return extractor


class NumericFeatureExtractor:
    """Extract and process numeric features from preprocessed data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'quantity_value',
            'pack_size',
            'has_organic',
            'has_gluten_free',
            'has_vegan',
            'has_non_gmo',
            'has_kosher',
            'is_gift',
            'is_bulk',
            'text_length',
            'num_bullet_points',
        ]
        self.is_fitted = False
    
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist in dataframe."""
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        return df
    
    def fit(self, df: pd.DataFrame):
        """Fit the scaler on training data."""
        df = self._ensure_columns(df.copy())
        features = df[self.feature_columns].fillna(0).values
        self.scaler.fit(features)
        self.is_fitted = True
        print(f"✅ Numeric feature scaler fitted on {len(self.feature_columns)} features")
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform numeric features."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted! Call fit() first.")
        
        df = self._ensure_columns(df.copy())
        features = df[self.feature_columns].fillna(0).values
        return self.scaler.transform(features)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def save(self, filepath: str = SCALER_FILE):
        """Save the scaler to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'is_fitted': self.is_fitted
            }, f)
        print(f"✅ Numeric scaler saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = SCALER_FILE) -> 'NumericFeatureExtractor':
        """Load a fitted scaler from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls()
        extractor.scaler = data['scaler']
        extractor.feature_columns = data['feature_columns']
        extractor.is_fitted = data['is_fitted']
        
        print(f"✅ Numeric scaler loaded from {filepath}")
        return extractor


class FeatureEngineer:
    """Main class that combines all feature extraction methods."""
    
    def __init__(self, 
                 max_tfidf_features: int = MODEL_CONFIG['tfidf_max_features'],
                 ngram_range: Tuple = MODEL_CONFIG['tfidf_ngram_range']):
        
        self.text_extractor = TextFeatureExtractor(
            max_features=max_tfidf_features,
            ngram_range=ngram_range
        )
        self.numeric_extractor = NumericFeatureExtractor()
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame):
        """Fit all feature extractors."""
        print("=" * 50)
        print("🔧 Fitting Feature Engineer")
        print("=" * 50)
        
        # Fit text extractor
        text_col = df['clean_text'] if 'clean_text' in df.columns else df['catalog_content']
        unit_col = df['quantity_unit'] if 'quantity_unit' in df.columns else None
        self.text_extractor.fit(text_col, unit_col)
        
        # Fit numeric extractor
        self.numeric_extractor.fit(df)
        
        self.is_fitted = True
        print("=" * 50)
        print("✅ Feature Engineer fitting complete!")
        print("=" * 50)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe to feature matrix."""
        if not self.is_fitted:
            raise ValueError("Feature Engineer not fitted! Call fit() first.")
        
        # Extract text features
        text_col = df['clean_text'] if 'clean_text' in df.columns else df['catalog_content']
        unit_col = df['quantity_unit'] if 'quantity_unit' in df.columns else None
        text_features = self.text_extractor.transform(text_col, unit_col)
        
        # Extract numeric features
        numeric_features = self.numeric_extractor.transform(df)
        
        # Combine all features
        all_features = np.hstack([text_features, numeric_features])
        
        print(f"📊 Feature matrix shape: {all_features.shape}")
        return all_features
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def save(self, text_path: str = TEXT_VECTORIZER_FILE, scaler_path: str = SCALER_FILE):
        """Save all extractors to disk."""
        self.text_extractor.save(text_path)
        self.numeric_extractor.save(scaler_path)
        print("✅ All feature extractors saved!")
    
    @classmethod
    def load(cls, text_path: str = TEXT_VECTORIZER_FILE, scaler_path: str = SCALER_FILE) -> 'FeatureEngineer':
        """Load all extractors from disk."""
        engineer = cls()
        engineer.text_extractor = TextFeatureExtractor.load(text_path)
        engineer.numeric_extractor = NumericFeatureExtractor.load(scaler_path)
        engineer.is_fitted = True
        return engineer


def extract_price_features_from_text(text: str) -> dict:
    """
    Extract potential price-related features from text.
    """
    features = {
        'mentioned_price_indicators': 0,
        'premium_indicators': 0,
        'budget_indicators': 0,
        'luxury_indicators': 0,
    }
    
    if not isinstance(text, str):
        return features
    
    text_lower = text.lower()
    
    # Premium/luxury indicators
    premium_words = ['premium', 'gourmet', 'luxury', 'artisan', 'handcrafted', 
                     'imported', 'exclusive', 'limited edition', 'deluxe']
    features['premium_indicators'] = sum(1 for word in premium_words if word in text_lower)
    
    # Budget indicators
    budget_words = ['value', 'economy', 'budget', 'affordable', 'cheap', 'discount']
    features['budget_indicators'] = sum(1 for word in budget_words if word in text_lower)
    
    # Luxury food indicators
    luxury_words = ['truffle', 'caviar', 'wagyu', 'saffron', 'gold', 'aged']
    features['luxury_indicators'] = sum(1 for word in luxury_words if word in text_lower)
    
    return features


if __name__ == "__main__":
    # Test feature extraction
    from config import SAMPLE_TEST_FILE
    from data_preprocessing import preprocess_dataframe
    
    print("\n🧪 Testing Feature Extraction")
    print("=" * 50)
    
    # Load and preprocess sample data
    sample_df = pd.read_csv(SAMPLE_TEST_FILE)
    processed_df = preprocess_dataframe(sample_df)
    
    # Test feature engineering
    engineer = FeatureEngineer(max_tfidf_features=1000)
    features = engineer.fit_transform(processed_df)
    
    print(f"\n📊 Final feature matrix shape: {features.shape}")
    print(f"📊 Sample features (first 5 rows, first 10 cols):\n{features[:5, :10]}")
