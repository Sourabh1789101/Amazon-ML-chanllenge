"""
Data Preprocessing Module for Amazon ML Challenge
Handles text cleaning, feature extraction from catalog content
"""
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class TextPreprocessor:
    """Preprocess and clean text data from catalog content."""
    
    def __init__(self):
        self.unit_mapping = {
            'ounce': 'oz', 'ounces': 'oz', 'fl oz': 'oz', 'fluid ounce': 'oz',
            'pound': 'lb', 'pounds': 'lb', 'lbs': 'lb',
            'gram': 'g', 'grams': 'g', 'gm': 'g',
            'kilogram': 'kg', 'kilograms': 'kg',
            'liter': 'l', 'liters': 'l', 'litre': 'l', 'litres': 'l',
            'milliliter': 'ml', 'milliliters': 'ml', 'ml': 'ml',
            'count': 'count', 'ct': 'count', 'piece': 'count', 'pieces': 'count',
            'pack': 'pack', 'packs': 'pack', 'pk': 'pack',
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\-\$\%]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_item_name(self, catalog_content: str) -> str:
        """Extract item name from catalog content."""
        if not isinstance(catalog_content, str):
            return ""
        
        match = re.search(r'Item Name:\s*(.+?)(?:Bullet Point|Product Description|Value:|$)', 
                         catalog_content, re.IGNORECASE | re.DOTALL)
        if match:
            return self.clean_text(match.group(1))
        return ""
    
    def extract_product_description(self, catalog_content: str) -> str:
        """Extract product description from catalog content."""
        if not isinstance(catalog_content, str):
            return ""
        
        match = re.search(r'Product Description:\s*(.+?)(?:Value:|Unit:|$)', 
                         catalog_content, re.IGNORECASE | re.DOTALL)
        if match:
            return self.clean_text(match.group(1))
        return ""
    
    def extract_bullet_points(self, catalog_content: str) -> List[str]:
        """Extract all bullet points from catalog content."""
        if not isinstance(catalog_content, str):
            return []
        
        bullet_points = re.findall(r'Bullet Point \d+:\s*(.+?)(?:Bullet Point|Product Description|Value:|$)', 
                                   catalog_content, re.IGNORECASE | re.DOTALL)
        return [self.clean_text(bp) for bp in bullet_points]
    
    def extract_quantity_info(self, catalog_content: str) -> Tuple[float, str]:
        """Extract quantity value and unit from catalog content."""
        if not isinstance(catalog_content, str):
            return 1.0, "count"
        
        # Extract Value
        value_match = re.search(r'Value:\s*([\d\.]+)', catalog_content)
        value = float(value_match.group(1)) if value_match else 1.0
        
        # Extract Unit
        unit_match = re.search(r'Unit:\s*(\w+)', catalog_content)
        unit = unit_match.group(1).lower() if unit_match else "count"
        
        # Normalize unit
        unit = self.unit_mapping.get(unit, unit)
        
        return value, unit
    
    def extract_brand(self, text: str) -> str:
        """Try to extract brand name from the beginning of item name."""
        if not isinstance(text, str):
            return ""
        
        # Common patterns: "Brand Name Product..." or first 1-2 words
        words = text.split()
        if len(words) >= 2:
            # Check if first words might be a brand (typically capitalized in original)
            potential_brand = ' '.join(words[:2])
            return potential_brand
        return words[0] if words else ""
    
    def extract_pack_size(self, text: str) -> int:
        """Extract pack/case size from text."""
        if not isinstance(text, str):
            return 1
        
        # Patterns like "case of 12", "pack of 6", "12 ct", "12 count"
        patterns = [
            r'case of (\d+)',
            r'pack of (\d+)',
            r'(\d+)\s*(?:ct|count|pk|pack)(?:\s|$|,)',
            r'(\d+)\s*x\s*\d+',  # 12 x 6oz
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return 1
    
    def extract_numeric_features(self, text: str) -> Dict[str, float]:
        """Extract numeric values from text like weights, counts."""
        features = {
            'has_organic': 0,
            'has_gluten_free': 0,
            'has_vegan': 0,
            'has_non_gmo': 0,
            'has_kosher': 0,
            'is_gift': 0,
            'is_bulk': 0,
            'text_length': 0,
            'num_bullet_points': 0,
        }
        
        if not isinstance(text, str):
            return features
        
        text_lower = text.lower()
        
        # Check for certifications/attributes
        features['has_organic'] = 1 if 'organic' in text_lower else 0
        features['has_gluten_free'] = 1 if 'gluten free' in text_lower or 'gluten-free' in text_lower else 0
        features['has_vegan'] = 1 if 'vegan' in text_lower else 0
        features['has_non_gmo'] = 1 if 'non-gmo' in text_lower or 'non gmo' in text_lower else 0
        features['has_kosher'] = 1 if 'kosher' in text_lower else 0
        features['is_gift'] = 1 if 'gift' in text_lower else 0
        features['is_bulk'] = 1 if any(word in text_lower for word in ['bulk', 'case of', 'pack of', 'wholesale']) else 0
        features['text_length'] = len(text)
        features['num_bullet_points'] = len(re.findall(r'Bullet Point', text))
        
        return features
    
    def process_catalog_content(self, catalog_content: str) -> Dict:
        """Process entire catalog content and extract all features."""
        result = {
            'item_name': self.extract_item_name(catalog_content),
            'description': self.extract_product_description(catalog_content),
            'bullet_points': self.extract_bullet_points(catalog_content),
            'clean_text': self.clean_text(catalog_content),
        }
        
        # Extract quantity
        value, unit = self.extract_quantity_info(catalog_content)
        result['quantity_value'] = value
        result['quantity_unit'] = unit
        
        # Extract additional features
        result['brand'] = self.extract_brand(result['item_name'])
        result['pack_size'] = self.extract_pack_size(catalog_content)
        
        # Extract numeric features
        numeric_features = self.extract_numeric_features(catalog_content)
        result.update(numeric_features)
        
        return result


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess entire dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'catalog_content' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional preprocessed features
    """
    preprocessor = TextPreprocessor()
    
    print("🔄 Processing catalog content...")
    
    # Process each row
    processed_features = df['catalog_content'].apply(preprocessor.process_catalog_content)
    
    # Convert to dataframe
    features_df = pd.DataFrame(processed_features.tolist())
    
    # Combine with original data
    result_df = pd.concat([df, features_df], axis=1)
    
    print(f"✅ Preprocessing complete! Added {len(features_df.columns)} new features.")
    
    return result_df


def load_and_preprocess_data(train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load and preprocess training and test data.
    
    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str, optional
        Path to test CSV file
        
    Returns:
    --------
    Tuple of preprocessed DataFrames (train, test)
    """
    print(f"📂 Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"   Training samples: {len(train_df)}")
    
    train_df = preprocess_dataframe(train_df)
    
    test_df = None
    if test_path:
        print(f"📂 Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        print(f"   Test samples: {len(test_df)}")
        test_df = preprocess_dataframe(test_df)
    
    return train_df, test_df


if __name__ == "__main__":
    # Test preprocessing
    import os
    from config import SAMPLE_TEST_FILE
    
    sample_df = pd.read_csv(SAMPLE_TEST_FILE)
    processed_df = preprocess_dataframe(sample_df)
    
    print("\n📊 Sample processed data:")
    print(processed_df[['sample_id', 'item_name', 'quantity_value', 'quantity_unit', 'pack_size']].head())
    print("\n📊 Numeric features:")
    print(processed_df[['has_organic', 'has_gluten_free', 'is_gift', 'text_length']].head())
